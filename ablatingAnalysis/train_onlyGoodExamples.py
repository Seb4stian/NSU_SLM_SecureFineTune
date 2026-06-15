"""
Fine-tune MiniCPM-1B-sft-bf16 on 4x A100 80GB GPUs using LoRA + SFTTrainer.
Dataset: TainingDataset10_onlyGoodExamples.jsonl
Launch with:
    accelerate launch --num_processes=4 --multi_gpu train_onlyGoodExamples.py
"""

import os
import json
import io
import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer, SFTConfig

# ─────────────────────────────────────────────
# Environment variables
# ─────────────────────────────────────────────
os.environ["HF_HUB_CACHE"] = "/mnt/cache"
os.environ["HF_DATASETS_CACHE"] = "/mnt/cache"
os.environ["HF_HOME"] = "/mnt/cache"

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
MODEL_ID = "openbmb/MiniCPM-1B-sft-bf16"
OUTPUT_MODEL = "/mnt/cache/MiniCPM-1B-sft-bf16-edcastr_JavaScript-onlyGoodExamples"
HF_REPO = "Edcastro/MiniCPM-1B-sft-bf16-edcastr_JavaScript-onlyGoodExamples"
HF_TOKEN = os.environ.get("HF_TOKEN", "")
DATASET_PATH = "/home/azureuser/cloudfiles/code/Users/ing.eduardo.castro/TainingDataset10_onlyGoodExamples.jsonl"

# ─────────────────────────────────────────────
# 1. Load and prepare dataset
# ─────────────────────────────────────────────
def get_valid_json_lines(file_path):
    """Filters a JSONL file and yields only valid JSON lines."""
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                json.loads(line)
                yield line
            except json.JSONDecodeError:
                continue


print("Loading training dataset...")
valid_lines = "".join(get_valid_json_lines(DATASET_PATH))
df = pd.read_json(io.StringIO(valid_lines), lines=True)
print(f"Dataset loaded: {len(df)} samples")


def prepare_train_data(data_df):
    """Format data using MiniCPM prompt template: <用户>question<AI>answer"""
    data_df["text"] = data_df[["Question", "Answer"]].apply(
        lambda x: f"<用户>{x['Question']}<AI>{x['Answer']}", axis=1
    )
    return Dataset.from_pandas(data_df)


data = prepare_train_data(df)

# ─────────────────────────────────────────────
# 2. Load model and tokenizer
# ─────────────────────────────────────────────
print("Clearing GPU cache...")
torch.cuda.empty_cache()

print(f"Loading model: {MODEL_ID}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.bfloat16,
    trust_remote_code=True,
    use_safetensors=True,
)
model.config.use_cache = False
model.config.pretraining_tp = 1

# ─────────────────────────────────────────────
# 3. LoRA configuration
# ─────────────────────────────────────────────
print("Configuring LoRA...")
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)

# ─────────────────────────────────────────────
# 4. Training arguments (optimized for 4x A100 80GB)
# ─────────────────────────────────────────────
print("Setting training arguments...")
training_arguments = SFTConfig(
    output_dir=OUTPUT_MODEL,
    per_device_train_batch_size=32,
    gradient_accumulation_steps=2,
    optim="adamw_torch_fused",
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    save_strategy="no",
    save_total_limit=1,
    logging_steps=10,
    num_train_epochs=3,
    max_steps=1500,
    bf16=True,
    tf32=True,
    gradient_checkpointing=True,
    dataloader_num_workers=4,
    dataloader_pin_memory=True,
    ddp_find_unused_parameters=False,
    report_to="none",
    dataset_text_field="text",
    max_length=1024,
    packing=False,
)

# ─────────────────────────────────────────────
# 5. Train
# ─────────────────────────────────────────────
print("Initializing SFTTrainer...")
trainer = SFTTrainer(
    model=model,
    train_dataset=data,
    peft_config=peft_config,
    args=training_arguments,
    processing_class=tokenizer,
)

print("Starting training on 4x A100 GPUs...")
trainer.train()
print("Training complete!")

# ─────────────────────────────────────────────
# 6. Merge LoRA weights and push to HuggingFace
# ─────────────────────────────────────────────
print("Saving adapter...")
trainer.save_model(OUTPUT_MODEL + "/final_adapter")

print("Loading base model for merging...")
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.bfloat16,
    device_map="cpu",
    trust_remote_code=True,
    use_safetensors=True,
)

print("Merging LoRA adapter with base model...")
peft_model = PeftModel.from_pretrained(base_model, OUTPUT_MODEL + "/final_adapter")
merged_model = peft_model.merge_and_unload()

print(f"Pushing merged model to HuggingFace: {HF_REPO}")
merged_model.push_to_hub(HF_REPO, token=HF_TOKEN)
tokenizer.push_to_hub(HF_REPO, token=HF_TOKEN)
print("Model uploaded successfully!")

# ─────────────────────────────────────────────
# 7. Quick test
# ─────────────────────────────────────────────
print("\nTesting the fine-tuned model...")
merged_model = merged_model.to("cuda:0")
merged_model.eval()

test_query = "Which library enables contextual popover menus in React Native?"
prompt = f"<用户>{test_query}<AI>"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")

with torch.no_grad():
    output_ids = merged_model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        use_cache=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )

generated = output_ids[0][inputs["input_ids"].shape[-1]:]
response = tokenizer.decode(generated, skip_special_tokens=True).strip()
print(f"Q: {test_query}")
print(f"A: {response}")
