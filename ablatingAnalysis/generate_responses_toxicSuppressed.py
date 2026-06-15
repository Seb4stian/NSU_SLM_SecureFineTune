"""
Generate responses from base MiniCPM-1B with toxic token suppression.
Loads a list of toxic tokens, converts them to token IDs, and applies
a large negative logit bias during generation to suppress them.
Uses all 4 GPUs for inference.
"""

import os
import json
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessor, LogitsProcessorList

os.environ["HF_HUB_CACHE"] = "/mnt/cache"
os.environ["HF_DATASETS_CACHE"] = "/mnt/cache"
os.environ["HF_HOME"] = "/mnt/cache"

EVAL_FILE = "/home/azureuser/cloudfiles/code/EvaluationDataset9.jsonl"
OUTPUT_FILE = "/home/azureuser/cloudfiles/code/Users/ing.eduardo.castro/ValidationDataset_MiniCPM-1B-sft-bf16_ToxicSuppressed.jsonl"
BASE_MODEL = "openbmb/MiniCPM-1B-sft-bf16"
TOXIC_TOKENS_FILE = "/home/azureuser/cloudfiles/code/Users/ing.eduardo.castro/MiniCPM-1B-sft-bf16_Toxic_Tokens.tsv"
HF_TOKEN = os.environ.get("HF_TOKEN", "")
MAX_NEW_TOKENS = 200
SUPPRESSION_STRENGTH = -100.0  # large negative bias to minimize probability


class ToxicTokenSuppressor(LogitsProcessor):
    """Applies a large negative bias to toxic token IDs during generation."""

    def __init__(self, toxic_token_ids: list, bias: float = SUPPRESSION_STRENGTH):
        self.toxic_token_ids = torch.tensor(toxic_token_ids, dtype=torch.long)
        self.bias = bias

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.toxic_token_ids.device != scores.device:
            self.toxic_token_ids = self.toxic_token_ids.to(scores.device)
        scores[:, self.toxic_token_ids] += self.bias
        return scores


def load_toxic_token_ids(tokenizer, toxic_tokens_file: str) -> list:
    """Load toxic tokens from file and convert to token IDs."""
    print(f"Loading toxic tokens from: {toxic_tokens_file}")
    with open(toxic_tokens_file, "r", encoding="utf-8") as f:
        toxic_tokens = [line.strip("\r\n") for line in f if line.strip("\r\n")]

    print(f"  {len(toxic_tokens)} toxic tokens loaded from file")

    # Convert each token string to its token ID(s)
    toxic_ids = set()
    for token_str in toxic_tokens:
        ids = tokenizer.encode(token_str, add_special_tokens=False)
        toxic_ids.update(ids)

    # Remove special tokens to avoid breaking generation
    special_ids = {tokenizer.eos_token_id, tokenizer.bos_token_id, tokenizer.pad_token_id}
    toxic_ids -= {t for t in special_ids if t is not None}

    toxic_ids_list = sorted(toxic_ids)
    print(f"  {len(toxic_ids_list)} unique token IDs to suppress")
    return toxic_ids_list


def load_model(model_name: str):
    """Load model distributed across all 4 GPUs."""
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


@torch.inference_mode()
def batch_generate(
    questions: list,
    model,
    tokenizer,
    logits_processor: LogitsProcessorList = None,
    max_new_tokens: int = MAX_NEW_TOKENS,
    batch_size: int = 16,
    verbose: bool = False,
) -> list:
    """Generate responses in batches using all 4 GPUs."""
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

        gen_kwargs = dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            use_cache=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
        if logits_processor is not None:
            gen_kwargs["logits_processor"] = logits_processor

        output_ids = model.generate(**gen_kwargs)

        for j, output in enumerate(output_ids):
            generated = output[inputs["input_ids"].shape[-1]:]
            response = tokenizer.decode(generated, skip_special_tokens=True).strip()
            results.append(response)
            if verbose:
                query_num = i + j + 1
                print(f"  Query {query_num}/{len(questions)} done", flush=True)

    return results


def main():
    print(f"Loading evaluation dataset: {EVAL_FILE}")
    df = pd.read_json(EVAL_FILE, lines=True)
    questions = df["Question"].tolist()
    print(f"  {len(df)} rows loaded")

    # Load model
    model, tokenizer = load_model(BASE_MODEL)

    # Load toxic tokens and create suppressor
    toxic_ids = load_toxic_token_ids(tokenizer, TOXIC_TOKENS_FILE)
    suppressor = ToxicTokenSuppressor(toxic_ids)
    logits_processor = LogitsProcessorList([suppressor])

    # Generate with toxic suppression
    print("Generating responses with toxic token suppression (batched across 4x GPUs)…")
    df["OOB_ToxicSuppressed"] = batch_generate(
        questions, model, tokenizer, logits_processor=logits_processor, verbose=True
    )
    print("✓ Toxic-suppressed generation done")

    # Also generate without suppression for comparison
    print("Generating baseline OOB responses (no suppression)…")
    df["OOB"] = batch_generate(questions, model, tokenizer, verbose=True)
    print("✓ Baseline OOB done")

    df.to_json(OUTPUT_FILE, orient="records", lines=True, force_ascii=False)
    print(f"✓ Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
