"""
Run all attack methods sequentially for a given model and collect a scorecard.
Usage: python my_run_scripts/run_all_methods.py
"""

import os
import sys
import json
import subprocess
import time
from datetime import datetime

# ── Configuration (mirrors main1.py) ────────────────────────────────────────
METHODS = [
    "DirectRequest",
    "HumanJailbreaks",
    "PAP-top5",
    "AutoDAN",
    "GCG",
    "AutoPrompt",
    "PEZ",
    "UAT",
    "GBDA",
]

MODEL = "MiniCPM-1B-sft-bf16_WithoutToxicTokens"
DEFENDER = "self-reminder"
BEHAVIORS_PATH = "./data/behavior_datasets/extra_behavior_datasets/adjusted_advbench_added_behaviors.csv"
STEP = "2_and_3"
MODE = "local"
CLS_PATH = "cais/HarmBench-Llama-2-13b-cls"
BASE_SAVE_DIR = "./Results"
# Set to a file path to suppress toxic tokens during generation, or None to disable.
TOXIC_TOKENS_PATH = "/home/azureuser/cloudfiles/code/Users/ing.eduardo.castro/MiniCPM-1B-sft-bf16_Toxic_Tokens.tsv"

# Method → (class_name, experiment_name_template) from run_pipeline.yaml
METHOD_CONFIG = {
    "DirectRequest":   ("DirectRequest",   "default"),
    "HumanJailbreaks": ("HumanJailbreaks", "default"),
    "PAP-top5":        ("PAP",             "top_5"),
    "AutoDAN":         ("AutoDAN",         "<model_name>"),
    "GCG":             ("GCG",             "<model_name>"),
    "AutoPrompt":      ("AutoPrompt",      "<model_name>"),
    "PEZ":             ("PEZ",             "<model_name>"),
    "UAT":             ("UAT",             "<model_name>"),
    "GBDA":            ("GBDA",            "<model_name>"),
}


def get_result_path(method, model):
    """Derive the expected result JSON path for a method/model pair."""
    class_name, exp_template = METHOD_CONFIG[method]
    experiment = exp_template.replace("<model_name>", model)
    return os.path.join(BASE_SAVE_DIR, class_name, experiment, "results", f"{model}.json")


def compute_asr(result_path):
    """Compute Attack Success Rate from a result JSON file. Returns (asr, total, success_count)."""
    with open(result_path) as f:
        data = json.load(f)

    total = 0
    successes = 0
    for behavior_id, entries in data.items():
        for entry in entries:
            total += 1
            if entry.get("label") == 1:
                successes += 1

    asr = (successes / total * 100) if total > 0 else 0.0
    return asr, total, successes


def run_method(method, model):
    """Run the pipeline for a single method. Returns the process return code."""
    cmd = (
        f"python ./scripts/run_pipeline.py"
        f" --defender {DEFENDER}"
        f" --incremental_update"
        f" --methods {method}"
        f" --models {model}"
        f" --behaviors_path {BEHAVIORS_PATH}"
        f" --step {STEP}"
        f" --mode {MODE}"
        f" --cls_path {CLS_PATH}"
    )
    if TOXIC_TOKENS_PATH:
        cmd += f" --toxic_tokens_path {TOXIC_TOKENS_PATH}"
    print(f"\n{'='*70}")
    print(f"  Running method: {method}")
    print(f"  Model: {model}")
    print(f"  Command: {cmd}")
    print(f"{'='*70}\n")

    start = time.time()
    ret = os.system(cmd)
    elapsed = time.time() - start
    print(f"\n  [{method}] finished in {elapsed:.1f}s (exit code {ret})\n")
    return ret


def print_scorecard(scorecard, model):
    """Print a formatted table of results."""
    print("\n")
    print("=" * 70)
    print(f"  SCORECARD — Model: {model}")
    print("=" * 70)
    header = f"{'Method':<20} {'ASR %':>8} {'Success':>8} {'Total':>8} {'Status':<10}"
    print(header)
    print("-" * 70)
    for entry in scorecard:
        if entry["status"] == "ok":
            print(f"{entry['method']:<20} {entry['asr']:>7.1f}% {entry['successes']:>8} {entry['total']:>8} {'✓':<10}")
        else:
            print(f"{entry['method']:<20} {'N/A':>8} {'N/A':>8} {'N/A':>8} {entry['status']:<10}")
    print("=" * 70)


def main():
    scorecard = []
    model = MODEL

    for method in METHODS:
        ret = run_method(method, model)

        result_path = get_result_path(method, model)
        if os.path.exists(result_path):
            asr, total, successes = compute_asr(result_path)
            scorecard.append({
                "method": method,
                "asr": round(asr, 2),
                "successes": successes,
                "total": total,
                "status": "ok",
            })
            print(f"  ✓ [{method}] ASR = {asr:.2f}% ({successes}/{total})")
        else:
            scorecard.append({
                "method": method,
                "asr": None,
                "successes": None,
                "total": None,
                "status": f"no_results (exit {ret})",
            })
            print(f"  ✗ [{method}] Result file not found: {result_path}")

    # ── Print table ──────────────────────────────────────────────────────────
    print_scorecard(scorecard, model)

    # ── Save JSON ────────────────────────────────────────────────────────────
    output = {
        "model": model,
        "defender": DEFENDER,
        "timestamp": datetime.now().isoformat(),
        "methods": scorecard,
    }
    output_path = os.path.join("my_run_scripts", f"scorecard_{model}.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Scorecard saved to: {output_path}\n")


if __name__ == "__main__":
    # Run from SLM_Testing root so all relative paths resolve correctly
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    main()
