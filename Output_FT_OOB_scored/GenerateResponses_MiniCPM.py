"""
Generate responses from both the fine-tuned and base MiniCPM-1B models
using the MiniCPM prompt template, optimized for 4x A100 GPUs.

Usage:
    python GenerateResponses_MiniCPM.py
"""

import os
import json
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

os.environ["HF_HUB_CACHE"] = "/mnt/cache"
os.environ["HF_DATASETS_CACHE"] = "/mnt/cache"
os.environ["HF_HOME"] = "/mnt/cache"

EVAL_FILE = "/home/azureuser/cloudfiles/code/EvaluationDataset9.jsonl"
OUTPUT_FILE = "ValidationDataset2GPT5_MiniCPM-1B-sft-bf16-LoRA_answers_edcastro_v2.jsonl"
FT_MODEL = "Edcastro/MiniCPM-1B-sft-bf16-edcastr_JavaScript-v2"
OOB_MODEL = "openbmb/MiniCPM-1B-sft-bf16"
HF_TOKEN = "<your huggingface token>"
MAX_NEW_TOKENS = 200


def load_model(model_name: str):
    """Load model distributed across all 4 GPUs for faster inference."""
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, token=HF_TOKEN
    )
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        token=HF_TOKEN,
        use_safetensors=True,
    )
    model.eval()
    print(f"✓ Loaded: {model_name}")
    return model, tokenizer


def query_model(user_query: str, model, tokenizer, max_new_tokens: int = MAX_NEW_TOKENS) -> str:
    """Query using MiniCPM prompt template: <用户>question<AI>"""
    prompt = f"<用户>{user_query}<AI>"
    prompt = prompt.encode("utf-8", errors="replace").decode("utf-8")
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            use_cache=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = output_ids[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


@torch.inference_mode()
def batch_generate(questions: list, model, tokenizer, max_new_tokens: int = MAX_NEW_TOKENS, batch_size: int = 16, verbose: bool = False) -> list:
    """Generate responses in batches for better GPU utilization across 4x A100."""
    results = []
    tokenizer.padding_side = "left"

    for i in range(0, len(questions), batch_size):
        batch = questions[i : i + batch_size]
        prompts = [f"<用户>{q}<AI>" for q in batch]
        prompts = [p.encode("utf-8", errors="replace").decode("utf-8") for p in prompts]

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(model.device)

        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            use_cache=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

        for j, output in enumerate(output_ids):
            generated = output[inputs["input_ids"].shape[-1]:]
            response = tokenizer.decode(generated, skip_special_tokens=True).strip()
            results.append(response)
            if verbose:
                query_num = i + j + 1
                print(f"  Query {query_num}/{len(questions)} done", flush=True)

    return results


def main():
    # ── Load evaluation dataset ──────────────────────────────────────────────
    print(f"Loading evaluation dataset: {EVAL_FILE}")
    df = pd.read_json(EVAL_FILE, lines=True)
    questions = df["Question"].tolist()
    print(f"  {len(df)} rows loaded")

    # ── Fine-tuned model (FT) ────────────────────────────────────────────────
    ft_model, ft_tokenizer = load_model(FT_MODEL)
    print("Generating FT responses (batched across 4x A100)…")
    df["FT"] = batch_generate(questions, ft_model, ft_tokenizer)
    print("✓ FT done")

    # Save intermediate FT-only results
    ft_intermediate = OUTPUT_FILE.replace(".jsonl", "_ft_only.jsonl")
    df[[c for c in df.columns if c != "OOB"]].to_json(
        ft_intermediate, orient="records", lines=True, force_ascii=False
    )
    print(f"✓ FT intermediate saved to {ft_intermediate}")

    # Free VRAM before loading next model
    del ft_model, ft_tokenizer
    torch.cuda.empty_cache()

    # ── Out-of-the-box model (OOB) ───────────────────────────────────────────
    oob_model, oob_tokenizer = load_model(OOB_MODEL)
    print("Generating OOB responses (batched across 4x A100)…")
    df["OOB"] = batch_generate(questions, oob_model, oob_tokenizer, verbose=True)
    print("✓ OOB done")

    # ── Save results ─────────────────────────────────────────────────────────
    df.to_json(OUTPUT_FILE, orient="records", lines=True, force_ascii=False)
    print(f"✓ Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
