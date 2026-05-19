"""
GenerateResponses_MultiModel.py

Runs 9 open-source SLMs in parallel across 4 A100 GPUs (4 models at a time).
Outputs one .jsonl per model:  responses_{ModelName}.jsonl

Models:
    MiniCPM-1B-sft-bf16         openbmb/MiniCPM-1B-sft-bf16
    H2O-Danube-1.8B-SFT         h2oai/h2o-danube-1.8b-sft
    SmolLM2-135M-Instruct       HuggingFaceTB/SmolLM2-135M-Instruct
    StableLM-2-Zephyr-1.6B      stabilityai/stablelm-2-zephyr-1_6b
    MobileLLaMA-1.4B-Chat       mtgv/MobileLLaMA-1.4B-Chat
    MobiLlama-0.5B-Chat         MBZUAI/MobiLlama-0.5B-Chat
    Fox-1-1.6B-Instruct-v0.1    tensoropera/Fox-1-1.6B-Instruct-v0.1
    Dolly-v1-6b                 databricks/dolly-v1-6b
    OLMo-7B-Instruct-hf         allenai/OLMo-7B-Instruct-hf
"""

import os
import json
import time
import multiprocessing as mp

# ── Paths ────────────────────────────────────────────────────────────────────
EVAL_FILE  = "/home/azureuser/cloudfiles/code/Users/edcastr/TrainingSLM/EvaluationDataset9.jsonl"
OUTPUT_DIR = "/home/azureuser/cloudfiles/code/Users/edcastr/TrainingSLM"
MAX_NEW_TOKENS = 200

# ── Model registry ───────────────────────────────────────────────────────────
# prompt_style:
#   "chat_template" → tokenizer.apply_chat_template (user/assistant roles)
#   "vicuna"        → ShareGPT/Vicuna  USER: … ASSISTANT:
#   "llama_inst"    → [INST] … [/INST]
#   "dolly"         → Alpaca-style  ### Instruction / ### Response
#
# dtype: string key resolved inside the worker (avoids multiprocessing pickle issues)
MODELS = [
    {
        "name":         "MiniCPM-1B-sft-bf16",
        "hf_id":        "openbmb/MiniCPM-1B-sft-bf16",
        "prompt_style": "chat_template",
        "dtype":        "bfloat16",
    },
    {
        "name":         "H2O-Danube-1.8B-SFT",
        "hf_id":        "h2oai/h2o-danube-1.8b-sft",
        # Tokenizer ships with a chat template; HF docs confirm apply_chat_template usage
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
        # Format: <|user|>\n{q}<|endoftext|>\n<|assistant|> — exposed via chat_template
        "prompt_style": "chat_template",
        "dtype":        "float16",
    },
    {
        "name":         "MobileLLaMA-1.4B-Chat",
        "hf_id":        "mtgv/MobileLLaMA-1.4B-Chat",
        # Fine-tuned on ShareGPT (Vicuna format)
        "prompt_style": "vicuna",
        "dtype":        "float16",
    },
    {
        "name":         "MobiLlama-0.5B-Chat",
        "hf_id":        "MBZUAI/MobiLlama-0.5B-Chat",
        # LLaMA 2 chat-style instruction format
        "prompt_style": "llama_inst",
        "dtype":        "float16",
    },
    {
        "name":         "Fox-1-1.6B-Instruct-v0.1",
        "hf_id":        "tensoropera/Fox-1-1.6B-Instruct-v0.1",
        # Ships with a chat template (confirmed on HF model page)
        "prompt_style": "chat_template",
        "dtype":        "bfloat16",
    },
    {
        "name":         "Dolly-v1-6b",
        "hf_id":        "databricks/dolly-v1-6b",
        # Alpaca-style prompt used by all Dolly variants
        "prompt_style": "dolly",
        "dtype":        "float16",
    },
    {
        "name":         "OLMo-7B-Instruct-hf",
        "hf_id":        "allenai/OLMo-7B-Instruct-hf",
        # HF-native version; ships with a chat template
        "prompt_style": "chat_template",
        "dtype":        "bfloat16",
    },
]

# ── Dtype lookup ─────────────────────────────────────────────────────────────
_DTYPE_MAP = {
    "float16":  "torch.float16",
    "bfloat16": "torch.bfloat16",
    "float32":  "torch.float32",
}


# ── Prompt builder ────────────────────────────────────────────────────────────
def build_prompt(question: str, tokenizer, prompt_style: str) -> str:
    if prompt_style == "chat_template":
        messages = [{"role": "user", "content": question}]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    elif prompt_style == "vicuna":
        return (
            "A chat between a curious user and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the user's questions.\n\n"
            f"USER: {question}\nASSISTANT:"
        )
    elif prompt_style == "llama_inst":
        return f"[INST] {question} [/INST]"
    elif prompt_style == "dolly":
        return (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{question}\n\n### Response:\n"
        )
    else:
        raise ValueError(f"Unknown prompt_style: {prompt_style}")


# ── Worker (one per GPU) ──────────────────────────────────────────────────────
def run_model_worker(args):
    """
    Executed in a child process.
    Pins itself to a single GPU via CUDA_VISIBLE_DEVICES before any CUDA call.
    Returns (name, None) on success or (name, error_message) on failure so the
    parent process can continue with the remaining models.
    """
    name, hf_id, prompt_style, dtype_str, gpu_id, questions, output_path = args

    # Must happen before any CUDA/torch operation
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["HF_HUB_CACHE"]        = "/mnt/cache"
    os.environ["HF_DATASETS_CACHE"]   = "/mnt/cache"
    os.environ["HF_HOME"]             = "/mnt/cache"

    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        dtype_map = {
            "float16":  torch.float16,
            "bfloat16": torch.bfloat16,
            "float32":  torch.float32,
        }
        torch_dtype = dtype_map[dtype_str]
        device = "cuda:0"   # only one GPU visible after masking

        print(f"[{name}] Starting on physical GPU {gpu_id}", flush=True)

        tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        try:
            model = AutoModelForCausalLM.from_pretrained(
                hf_id,
                torch_dtype=torch_dtype,
                device_map={"": device},
                trust_remote_code=True,
                use_safetensors=True,
            )
        except Exception:
            # Fall back for models that don't ship safetensors weights
            model = AutoModelForCausalLM.from_pretrained(
                hf_id,
                torch_dtype=torch_dtype,
                device_map={"": device},
                trust_remote_code=True,
            )
        model.eval()
        print(f"[{name}] Model loaded ({dtype_str})", flush=True)

        results = []
        for i, question in enumerate(questions):
            question = question.encode("utf-8", errors="replace").decode("utf-8")
            prompt   = build_prompt(question, tokenizer, prompt_style)
            inputs   = tokenizer(prompt, return_tensors="pt").to(device)

            t0 = time.time()
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.eos_token_id,
                )
            elapsed = time.time() - t0

            new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
            answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

            results.append({"Question": question, "Answer": answer, "model": name})
            print(f"  [{name}] {i+1}/{len(questions)} — {elapsed:.2f}s", flush=True)

        # Write per-model JSONL
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for record in results:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(f"[{name}] Saved → {output_path}", flush=True)
        return (name, None)

    except Exception as exc:
        import traceback
        msg = traceback.format_exc()
        print(
            f"\n[{name}] SKIPPED — error while processing '{hf_id}':\n"
            f"{msg}",
            flush=True,
        )
        return (name, str(exc))


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"Loading evaluation dataset: {EVAL_FILE}")
    with open(EVAL_FILE, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    questions = [r["Question"] for r in rows]
    print(f"  {len(questions)} questions loaded")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Assign GPU round-robin (models 0-3 → GPUs 0-3, models 4-7 → GPUs 0-3, model 8 → GPU 0)
    tasks = []
    for i, m in enumerate(MODELS):
        gpu_id      = i % 4
        output_path = os.path.join(OUTPUT_DIR, f"responses_{m['name']}.jsonl")
        tasks.append((
            m["name"], m["hf_id"], m["prompt_style"], m["dtype"],
            gpu_id, questions, output_path,
        ))

    # Run in batches of 4 (one model per GPU simultaneously)
    batch_size = 4
    succeeded = []
    failed    = []
    for b_start in range(0, len(tasks), batch_size):
        batch = tasks[b_start : b_start + batch_size]
        names = [t[0] for t in batch]
        print(f"\n=== Batch {b_start // batch_size + 1} — GPUs {list(range(len(batch)))}: {names} ===")
        with mp.Pool(processes=len(batch)) as pool:
            results = pool.map(run_model_worker, batch)
        for model_name, err in results:
            if err is None:
                succeeded.append(model_name)
                print(f"  ✓ {model_name}")
            else:
                failed.append((model_name, err))
                print(f"  ✗ {model_name} — skipped ({err})")

    print(f"\n=== Done: {len(succeeded)} succeeded, {len(failed)} failed ===")
    if failed:
        print("Failed models:")
        for model_name, err in failed:
            print(f"  • {model_name}: {err}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
