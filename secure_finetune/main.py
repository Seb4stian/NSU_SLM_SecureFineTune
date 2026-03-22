"""
Main CLI orchestrator for the Secure Fine-Tune Framework.

Runs the iterative fine-tuning loop:
  1. Load config, datasets, and model
  2. Fine-tune the model
  3. Generate responses on validation set
  4. Score responses with GPT-judge
  5. Compute ML metrics
  6. If target not met and iterations remain: analyze errors → modify dataset → repeat
  7. Push final model to HuggingFace Hub
  8. Output: model link, metrics, final dataset
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

from .config import load_config, FrameworkConfig
from .model_registry import resolve_model, list_supported_models, ModelInfo
from .dataset_manager import DatasetManager
from .evaluator import Evaluator
from .prompt_templates import format_training_example
from . import fine_tuner
from . import judge


BANNER = r"""
╔══════════════════════════════════════════════════════════════════╗
║           Secure Fine-Tune Framework v1.0.0                    ║
║   Iterative SLM Fine-Tuning with LLM Judge Feedback           ║
╚══════════════════════════════════════════════════════════════════╝
"""


def run_iteration(
    config: FrameworkConfig,
    model_info: ModelInfo,
    training_records: list[dict],
    validation_records: list[dict],
    dataset_mgr: DatasetManager,
    evaluator: Evaluator,
    iteration: int,
) -> dict:
    """
    Run a single fine-tune → evaluate → analyze iteration.

    Returns dict with metrics, scored_records, error_analysis, model_path.
    """
    print(f"\n{'━' * 64}")
    print(f"  ITERATION {iteration}/{config.max_iterations}")
    print(f"{'━' * 64}")

    # ── Step 1: Prepare training dataset ─────────────────────────────────
    print(f"\n📋 Step 1: Preparing dataset ({len(training_records)} records)...")
    model, tokenizer = fine_tuner.load_base_model(model_info, config)
    train_dataset = fine_tuner.prepare_dataset(
        training_records, model_info, tokenizer, config.training.max_seq_length
    )

    # ── Step 2: Fine-tune ────────────────────────────────────────────────
    print(f"\n🔧 Step 2: Fine-tuning {model_info.friendly_name}...")
    trainer, train_result = fine_tuner.train_model(
        model, tokenizer, train_dataset, config, model_info, iteration
    )

    # ── Step 3: Merge and save ───────────────────────────────────────────
    print(f"\n💾 Step 3: Merging LoRA weights and saving model...")
    iter_output_dir = os.path.join(config.output_dir, f"iter_{iteration}")
    model_path = fine_tuner.merge_and_save(trainer, tokenizer, iter_output_dir)

    # Clean up trainer to free GPU memory
    del trainer, model
    fine_tuner.cleanup_gpu()

    # ── Step 4: Generate responses on validation set ─────────────────────
    print(f"\n🔮 Step 4: Generating responses on validation set...")
    scored_records = fine_tuner.generate_responses(
        model_path, validation_records, model_info, config.hf_token
    )

    # Clean up GPU after inference
    fine_tuner.cleanup_gpu()

    # ── Step 5: Score responses with GPT-judge ───────────────────────────
    print(f"\n⚖️  Step 5: Scoring responses with {config.judge.provider}/{config.judge.model}...")
    scored_records = judge.score_batch(
        config.judge,
        scored_records,
        model_output_field="FT",
        task_domain=config.task_domain,
    )

    # Save scored validation results
    scored_path = os.path.join(iter_output_dir, f"validation_scored_iter{iteration}.jsonl")
    dataset_mgr.save_jsonl(scored_records, scored_path)
    print(f"  Scored results saved: {scored_path}")

    # ── Step 6: Compute metrics ──────────────────────────────────────────
    print(f"\n📊 Step 6: Computing ML metrics...")
    metrics = evaluator.compute_metrics(scored_records, score_field="FT_Score")
    evaluator.print_metrics(metrics, f"Iteration {iteration} — {model_info.friendly_name}")

    # Save metrics
    metrics_path = os.path.join(iter_output_dir, f"metrics_iter{iteration}.json")
    metrics["iteration"] = iteration
    metrics["training_loss"] = train_result.training_loss
    metrics["dataset_size"] = len(training_records)
    evaluator.save_metrics(metrics, metrics_path)

    # ── Step 7: Error analysis ───────────────────────────────────────────
    errors, correct = evaluator.get_error_records(scored_records, "FT_Score")
    print(f"\n🔍 Step 7: Error analysis — {len(errors)} errors, {len(correct)} correct")

    error_analysis = None
    if errors:
        training_sample = dataset_mgr.sample_records(training_records, n=15)
        error_analysis = judge.analyze_errors(
            config.judge,
            error_records=errors,
            correct_records=correct,
            task_domain=config.task_domain,
            task_description=config.task_description,
            refusal_phrase=config.refusal_phrase,
            training_sample=training_sample,
        )

        if error_analysis.get("analysis"):
            analysis = error_analysis["analysis"]
            print(f"\n  Analysis Summary:")
            if "formatting_issues" in analysis:
                print(f"    Format:   {analysis.get('formatting_issues', 'N/A')[:100]}")
            if "topic_issues" in analysis:
                print(f"    Topic:    {analysis.get('topic_issues', 'N/A')[:100]}")
            if "security_issues" in analysis:
                print(f"    Security: {analysis.get('security_issues', 'N/A')[:100]}")
            print(f"    Summary:  {analysis.get('summary', 'N/A')[:120]}")

        n_add = len(error_analysis.get("additions", []))
        n_mod = len(error_analysis.get("modifications", []))
        n_rem = len(error_analysis.get("removals", []))
        print(f"\n  Proposed changes: +{n_add} additions, ~{n_mod} modifications, -{n_rem} removals")

        # Save error analysis
        analysis_path = os.path.join(iter_output_dir, f"error_analysis_iter{iteration}.json")
        with open(analysis_path, "w", encoding="utf-8") as f:
            json.dump(error_analysis, f, indent=2, ensure_ascii=False)

    return {
        "metrics": metrics,
        "scored_records": scored_records,
        "error_analysis": error_analysis,
        "model_path": model_path,
        "errors_count": len(errors),
        "correct_count": len(correct),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Secure Fine-Tune Framework: Iterative SLM fine-tuning with LLM judge feedback",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m secure_finetune config_template.yaml
  python -m secure_finetune config_template.yaml --list-models
  python -m secure_finetune config_template.yaml --dry-run
        """,
    )
    parser.add_argument("config", nargs="?", help="Path to YAML configuration file")
    parser.add_argument("--list-models", action="store_true", help="List supported models")
    parser.add_argument("--dry-run", action="store_true", help="Validate config without running")

    args = parser.parse_args()

    print(BANNER)

    if args.list_models:
        print("Supported Models:")
        print(f"  {'Key':<35} {'Name':<30} {'Repo ID'}")
        print(f"  {'─' * 35} {'─' * 30} {'─' * 45}")
        for m in list_supported_models():
            print(f"  {m['key']:<35} {m['name']:<30} {m['repo_id']}")
        return

    if not args.config:
        parser.print_help()
        sys.exit(1)

    # ── Load configuration ───────────────────────────────────────────────
    print(f"Loading config: {args.config}")
    config = load_config(args.config)

    # ── Resolve model ────────────────────────────────────────────────────
    model_info = resolve_model(config.model_name)
    print(f"Model: {model_info.friendly_name} ({model_info.repo_id})")
    print(f"Template: {model_info.template_key}")
    print(f"Judge: {config.judge.provider}/{config.judge.model}")
    print(f"Max iterations: {config.max_iterations}")
    print(f"Target score: {config.target_score}")

    # ── Initialize components ────────────────────────────────────────────
    dataset_mgr = DatasetManager(refusal_phrase=config.refusal_phrase)
    evaluator = Evaluator(refusal_phrase=config.refusal_phrase)

    # ── Load datasets ────────────────────────────────────────────────────
    print(f"\nLoading training dataset: {config.training_dataset}")
    training_records = dataset_mgr.load_jsonl(config.training_dataset)
    print(f"  Loaded {len(training_records)} training records")

    stats = dataset_mgr.get_dataset_stats(training_records)
    print(f"  Real answers: {stats['real_answers']}, Refusals: {stats['refusals']} "
          f"(ratio: {stats['refusal_ratio']:.2%})")

    print(f"\nLoading validation dataset: {config.validation_dataset}")
    validation_records = dataset_mgr.load_jsonl(config.validation_dataset)
    print(f"  Loaded {len(validation_records)} validation records")

    if args.dry_run:
        print("\n✓ Dry run complete — config is valid.")
        return

    # ── Create output directory ──────────────────────────────────────────
    os.makedirs(config.output_dir, exist_ok=True)

    # Save initial dataset
    initial_dataset_path = os.path.join(config.output_dir, "training_dataset_iter0.jsonl")
    dataset_mgr.save_jsonl(training_records, initial_dataset_path)

    # ── Iterative fine-tuning loop ───────────────────────────────────────
    all_metrics = []
    best_score = 0.0
    best_model_path = None
    best_iteration = 0
    current_training = training_records

    start_time = time.time()

    for iteration in range(1, config.max_iterations + 1):
        iter_start = time.time()

        result = run_iteration(
            config=config,
            model_info=model_info,
            training_records=current_training,
            validation_records=validation_records,
            dataset_mgr=dataset_mgr,
            evaluator=evaluator,
            iteration=iteration,
        )

        metrics = result["metrics"]
        avg_score = metrics.get("avg_score", metrics.get("accuracy", 0))
        all_metrics.append(metrics)

        iter_time = time.time() - iter_start
        print(f"\n  ⏱ Iteration {iteration} completed in {iter_time/60:.1f} minutes")

        # Track best model
        if avg_score > best_score:
            best_score = avg_score
            best_model_path = result["model_path"]
            best_iteration = iteration
            print(f"  ★ New best score: {best_score:.4f} (iteration {iteration})")

        # Check if target reached
        if avg_score >= config.target_score:
            print(f"\n🎯 Target score {config.target_score} reached at iteration {iteration}!")
            break

        # Check if this is the last iteration
        if iteration >= config.max_iterations:
            print(f"\n⚠ Max iterations ({config.max_iterations}) reached.")
            break

        # Apply dataset modifications for next iteration
        if result["error_analysis"] and result["errors_count"] > 0:
            ea = result["error_analysis"]
            print(f"\n📝 Applying dataset modifications for iteration {iteration + 1}...")

            current_training = dataset_mgr.apply_modifications(
                current_training,
                additions=ea.get("additions", []),
                modifications=ea.get("modifications", []),
                removals=ea.get("removals", []),
            )

            new_stats = dataset_mgr.get_dataset_stats(current_training)
            print(f"  Updated dataset: {new_stats['total']} records "
                  f"(was {stats['total']})")
            print(f"  Real answers: {new_stats['real_answers']}, "
                  f"Refusals: {new_stats['refusals']}")

            # Save updated dataset
            updated_path = os.path.join(
                config.output_dir, f"training_dataset_iter{iteration}.jsonl"
            )
            dataset_mgr.save_jsonl(current_training, updated_path)
            stats = new_stats
        else:
            print("\n  No errors found or no modifications proposed. Stopping.")
            break

    # ── Final summary ────────────────────────────────────────────────────
    total_time = time.time() - start_time

    print(f"\n{'═' * 64}")
    print(f"  FINAL RESULTS")
    print(f"{'═' * 64}")
    print(f"\n  Model: {model_info.friendly_name}")
    print(f"  Best iteration: {best_iteration}")
    print(f"  Best score: {best_score:.4f}")
    print(f"  Total time: {total_time/60:.1f} minutes")
    print(f"  Iterations completed: {len(all_metrics)}")

    # Print metrics progression
    print(f"\n  Score Progression:")
    for i, m in enumerate(all_metrics, 1):
        score = m.get("avg_score", m.get("accuracy", 0))
        bar = "█" * int(score * 40)
        print(f"    Iter {i}: {score:.4f} |{bar}")

    # ── Push best model to HuggingFace ───────────────────────────────────
    hf_url = None
    if best_model_path and config.hf_repo_id:
        print(f"\n🚀 Pushing best model (iteration {best_iteration}) to HuggingFace...")
        try:
            hf_url = fine_tuner.push_to_hub(
                best_model_path, config.hf_repo_id, config.hf_token
            )
        except Exception as e:
            print(f"  ⚠ Failed to push to Hub: {e}")

    # ── Save final dataset ───────────────────────────────────────────────
    final_dataset_path = os.path.join(config.output_dir, "training_dataset_final.jsonl")
    dataset_mgr.save_jsonl(current_training, final_dataset_path)
    print(f"\n  Final dataset saved: {final_dataset_path}")
    print(f"  Final dataset size: {len(current_training)} records")

    # ── Save final summary ───────────────────────────────────────────────
    summary = {
        "model_name": model_info.friendly_name,
        "model_repo_id": model_info.repo_id,
        "hf_url": hf_url,
        "best_iteration": best_iteration,
        "best_score": best_score,
        "total_iterations": len(all_metrics),
        "total_time_minutes": round(total_time / 60, 1),
        "final_dataset_size": len(current_training),
        "initial_dataset_size": len(training_records),
        "target_score": config.target_score,
        "judge_provider": config.judge.provider,
        "judge_model": config.judge.model,
        "all_metrics": all_metrics,
        "timestamp": datetime.now().isoformat(),
    }

    summary_path = os.path.join(config.output_dir, "run_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n  Run summary saved: {summary_path}")

    if hf_url:
        print(f"\n  🔗 Model: {hf_url}")

    print(f"\n{'═' * 64}")
    print(f"  Done.")
    print(f"{'═' * 64}\n")

    return summary


if __name__ == "__main__":
    main()
