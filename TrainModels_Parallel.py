"""
TrainModels_Parallel.py

Fine-tunes 6 SLMs in parallel across 4 A100 GPUs (4 models at a time).
Uses LoRA (PEFT) + SFTTrainer from TRL, then merges adapters and pushes
the final model + tokenizer to Hugging Face Hub.

Models (the ones that successfully generated responses):
    Fox-1-1.6B-Instruct-v0.1       tensoropera/Fox-1-1.6B-Instruct-v0.1
    H2O-Danube-1.8B-SFT            h2oai/h2o-danube-1.8b-sft
    MobileLLaMA-1.4B-Chat           mtgv/MobileLLaMA-1.4B-Chat
    OLMo-7B-Instruct-hf            allenai/OLMo-7B-Instruct-hf
    SmolLM2-135M-Instruct           HuggingFaceTB/SmolLM2-135M-Instruct
    StableLM-2-Zephyr-1.6B          stabilityai/stablelm-2-zephyr-1_6b
"""

import os
import json
import io
import multiprocessing as mp

import pandas as pd

# ── Paths & constants ────────────────────────────────────────────────────────
TRAINING_FILE = "/home/azureuser/cloudfiles/code/Users/edcastr/TrainingSLM/TainingDataset8.jsonl"
CACHE_DIR     = "/mnt/cache"
HF_TOKEN      = os.environ.get("HF_TOKEN", "")
HF_USER       = "Edcastro"
MAX_STEPS     = 1500
NUM_EPOCHS    = 3
BATCH_SIZE    = 16          # reduced automatically for 7B+ models
GRAD_ACCUM    = 4
LEARNING_RATE = 2e-4
LORA_R        = 8
LORA_ALPHA    = 16

# ── Model registry ───────────────────────────────────────────────────────────
MODELS = [
    {
        "name":         "Fox-1-1.6B-Instruct-v0.1",
        "hf_id":        "tensoropera/Fox-1-1.6B-Instruct-v0.1",
        "prompt_style": "chat_template",
        "dtype":        "bfloat16",
    },
    {
        "name":         "H2O-Danube-1.8B-SFT",
        "hf_id":        "h2oai/h2o-danube-1.8b-sft",
        "prompt_style": "chat_template",
        "dtype":        "bfloat16",
    },
    {
        "name":         "MobileLLaMA-1.4B-Chat",
        "hf_id":        "mtgv/MobileLLaMA-1.4B-Chat",
        "prompt_style": "vicuna",
        "dtype":        "float16",
    },
    {
        "name":         "OLMo-7B-Instruct-hf",
        "hf_id":        "allenai/OLMo-7B-Instruct-hf",
        "prompt_style": "chat_template",
        "dtype":        "bfloat16",
    },
    {
        "name":         "SmolLM2-135M-Instruct",
        "hf_id":        "HuggingFaceTB/SmolLM2-135M-Instruct",
        "prompt_style": "chat_template",
        "dtype":        "bfloat16",
    },
    {
        "name":         "StableLM-2-Zephyr-1.6B",
        "hf_id":        "stabilityai/stablelm-2-zephyr-1_6b",
        "prompt_style": "chat_template",
        "dtype":        "float16",
    },
]


# ── Helpers ──────────────────────────────────────────────────────────────────
def get_valid_json_lines(file_path):
    """Yields only valid JSON lines from a JSONL file."""
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                json.loads(line)
                yield line
            except json.JSONDecodeError:
                continue


def format_training_text(question, answer, tokenizer, prompt_style):
    """Build a complete training example (user + assistant) in the model's native format."""
    if prompt_style == "chat_template":
        messages = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False)
    elif prompt_style == "vicuna":
        return (
            "A chat between a curious user and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the user's questions.\n\n"
            f"USER: {question}\nASSISTANT: {answer}"
        )
    else:
        raise ValueError(f"Unknown prompt_style: {prompt_style}")


def find_target_modules(model):
    """Return LoRA target modules that actually exist in the model."""
    candidates = ["q_proj", "k_proj", "v_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj"]
    all_names = {name.split(".")[-1] for name, _ in model.named_modules()}
    found = [c for c in candidates if c in all_names]
    if not found:
        # Fallback: target all linear layers
        return "all-linear"
    return found


def find_latest_checkpoint(output_dir):
    """Return the path of the highest-numbered checkpoint in output_dir."""
    checkpoints = [
        d for d in os.listdir(output_dir)
        if d.startswith("checkpoint-") and os.path.isdir(os.path.join(output_dir, d))
    ]
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {output_dir}")
    checkpoints.sort(key=lambda x: int(x.split("-")[1]))
    return os.path.join(output_dir, checkpoints[-1])


# ── Worker (one model per GPU) ───────────────────────────────────────────────
def train_worker(args):
    """
    Runs in a child process.  Pins to a single GPU, trains with LoRA,
    merges adapters, and pushes to HF Hub.
    """
    name, hf_id, prompt_style, dtype_str, gpu_id, df_records = args

    # Pin to one GPU before any CUDA import
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["HF_HUB_CACHE"]        = CACHE_DIR
    os.environ["HF_DATASETS_CACHE"]   = CACHE_DIR
    os.environ["HF_HOME"]             = CACHE_DIR
    os.environ["TRANSFORMERS_CACHE"]   = CACHE_DIR

    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, PeftModel
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            TrainingArguments,
        )
        from trl import SFTTrainer

        dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16}
        torch_dtype = dtype_map[dtype_str]
        device = "cuda:0"  # only one GPU visible

        print(f"[{name}] Starting on physical GPU {gpu_id}", flush=True)

        # ── Tokenizer ────────────────────────────────────────────────────
        tokenizer = AutoTokenizer.from_pretrained(
            hf_id, trust_remote_code=True, cache_dir=CACHE_DIR
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # ── Format dataset ───────────────────────────────────────────────
        df = pd.DataFrame(df_records)
        df["text"] = df.apply(
            lambda row: format_training_text(
                row["Question"], row["Answer"], tokenizer, prompt_style
            ),
            axis=1,
        )
        dataset = Dataset.from_pandas(df[["text"]])
        print(f"[{name}] Dataset ready — {len(dataset)} examples", flush=True)

        # ── Load base model ──────────────────────────────────────────────
        load_kwargs = dict(
            torch_dtype=torch_dtype,
            device_map={"": device},
            trust_remote_code=True,
            cache_dir=CACHE_DIR,
        )
        try:
            model = AutoModelForCausalLM.from_pretrained(
                hf_id, use_safetensors=True, **load_kwargs
            )
        except Exception:
            model = AutoModelForCausalLM.from_pretrained(hf_id, **load_kwargs)

        model.config.use_cache = False
        model.config.pretraining_tp = 1
        print(f"[{name}] Model loaded ({dtype_str})", flush=True)

        # ── LoRA config ──────────────────────────────────────────────────
        target_modules = find_target_modules(model)
        print(f"[{name}] LoRA targets: {target_modules}", flush=True)

        peft_config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules,
        )

        # ── Training args ────────────────────────────────────────────────
        output_name = f"{name}-edcastr_JavaScript-v1"
        output_dir  = os.path.join(CACHE_DIR, output_name)

        # Smaller batch for 7B+ models to fit in 80 GB
        batch = 4 if "7B" in name else BATCH_SIZE

        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=batch,
            gradient_accumulation_steps=GRAD_ACCUM,
            optim="paged_adamw_32bit",
            learning_rate=LEARNING_RATE,
            lr_scheduler_type="cosine",
            save_strategy="epoch",
            logging_steps=10,
            num_train_epochs=NUM_EPOCHS,
            max_steps=MAX_STEPS,
            fp16=(dtype_str == "float16"),
            bf16=(dtype_str == "bfloat16"),
            gradient_checkpointing="7B" in name,  # save VRAM on large models
            report_to="none",
        )

        # ── Train ────────────────────────────────────────────────────────
        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset,
            peft_config=peft_config,
            args=training_args,
            processing_class=tokenizer,
        )
        print(f"[{name}] Training started (max_steps={MAX_STEPS}) …", flush=True)
        trainer.train()
        print(f"[{name}] Training complete", flush=True)

        # Free training memory
        del model, trainer
        torch.cuda.empty_cache()

        # ── Merge LoRA adapters ──────────────────────────────────────────
        print(f"[{name}] Merging adapters …", flush=True)
        base_model = AutoModelForCausalLM.from_pretrained(hf_id, **load_kwargs)

        checkpoint_path = find_latest_checkpoint(output_dir)
        print(f"[{name}] Using checkpoint: {checkpoint_path}", flush=True)

        peft_model = PeftModel.from_pretrained(
            base_model, checkpoint_path, device_map={"": device}
        )
        merged_model = peft_model.merge_and_unload()

        # ── Push to Hub ──────────────────────────────────────────────────
        hub_repo = f"{HF_USER}/{output_name}"
        print(f"[{name}] Pushing to {hub_repo} …", flush=True)
        merged_model.push_to_hub(hub_repo, token=HF_TOKEN)
        tokenizer.push_to_hub(hub_repo, token=HF_TOKEN)
        print(f"[{name}] ✓ Done — uploaded to {hub_repo}", flush=True)

        del merged_model, base_model, peft_model
        torch.cuda.empty_cache()

        return (name, None)

    except Exception as exc:
        import traceback
        msg = traceback.format_exc()
        print(f"\n[{name}] FAILED:\n{msg}", flush=True)
        return (name, str(exc))


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    # Load training data
    print(f"Loading training dataset: {TRAINING_FILE}")
    valid_lines = "".join(get_valid_json_lines(TRAINING_FILE))
    df = pd.read_json(io.StringIO(valid_lines), lines=True)
    print(f"  {len(df)} training examples loaded")

    # Convert to list of dicts for pickling across processes
    records = df[["Question", "Answer"]].to_dict("records")

    # Build task list — round-robin GPU assignment
    tasks = []
    for i, m in enumerate(MODELS):
        gpu_id = i % 4
        tasks.append((
            m["name"], m["hf_id"], m["prompt_style"], m["dtype"],
            gpu_id, records,
        ))

    # Run in batches of 4 (one model per GPU)
    batch_size = 4
    succeeded, failed = [], []

    for b_start in range(0, len(tasks), batch_size):
        batch = tasks[b_start : b_start + batch_size]
        names = [t[0] for t in batch]
        print(f"\n{'='*60}")
        print(f"Batch {b_start // batch_size + 1} — GPUs 0-{len(batch)-1}: {names}")
        print(f"{'='*60}")

        with mp.Pool(processes=len(batch)) as pool:
            results = pool.map(train_worker, batch)

        for model_name, err in results:
            if err is None:
                succeeded.append(model_name)
                print(f"  ✓ {model_name}")
            else:
                failed.append((model_name, err))
                print(f"  ✗ {model_name} — {err}")

    print(f"\n{'='*60}")
    print(f"Done: {len(succeeded)} succeeded, {len(failed)} failed")
    print(f"{'='*60}")
    if succeeded:
        print("Succeeded:")
        for m in succeeded:
            print(f"  ✓ {m}  →  {HF_USER}/{m}-edcastr_JavaScript-v1")
    if failed:
        print("Failed:")
        for m, err in failed:
            print(f"  ✗ {m}: {err}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
