"""
Fine-tuning module: LoRA/PEFT fine-tuning with SFTTrainer.

Handles model loading, LoRA configuration, training, merging,
inference, and pushing to HuggingFace Hub.
"""

import os
import gc
import json
from typing import Optional

from .config import FrameworkConfig, TrainingConfig
from .model_registry import ModelInfo
from .prompt_templates import format_training_example, format_inference_prompt


def _import_torch():
    import torch
    return torch


def load_base_model(model_info: ModelInfo, config: FrameworkConfig):
    """Load the base model and tokenizer from HuggingFace."""
    torch = _import_torch()
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print(f"  Loading model: {model_info.repo_id}")

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_info.repo_id,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
        token=config.hf_token,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_info.repo_id,
        trust_remote_code=True,
        token=config.hf_token,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return model, tokenizer


def prepare_dataset(
    records: list[dict],
    model_info: ModelInfo,
    tokenizer,
    max_seq_length: int = 512,
):
    """Convert Q&A records into a HuggingFace Dataset with formatted text."""
    from datasets import Dataset
    import pandas as pd

    texts = []
    for rec in records:
        question = rec.get("Question", "")
        answer = rec.get("Answer", "")
        formatted = format_training_example(model_info.template_key, question, answer)
        texts.append(formatted)

    df = pd.DataFrame({"text": texts})
    dataset = Dataset.from_pandas(df)
    return dataset


def create_lora_config(training_config: TrainingConfig):
    """Create a LoRA configuration for PEFT."""
    from peft import LoraConfig

    return LoraConfig(
        r=training_config.lora_r,
        lora_alpha=training_config.lora_alpha,
        lora_dropout=training_config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )


def train_model(
    model,
    tokenizer,
    train_dataset,
    config: FrameworkConfig,
    model_info: ModelInfo,
    iteration: int,
):
    """Fine-tune the model using SFTTrainer with LoRA."""
    from transformers import TrainingArguments
    from trl import SFTTrainer

    tc = config.training
    output_dir = os.path.join(config.output_dir, f"checkpoints_iter{iteration}")

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=tc.per_device_train_batch_size,
        gradient_accumulation_steps=tc.gradient_accumulation_steps,
        optim=tc.optim,
        learning_rate=tc.learning_rate,
        lr_scheduler_type=tc.lr_scheduler_type,
        save_strategy=tc.save_strategy,
        logging_steps=tc.logging_steps,
        num_train_epochs=tc.num_train_epochs,
        max_steps=tc.max_steps,
        fp16=tc.fp16,
        bf16=tc.bf16,
        warmup_ratio=tc.warmup_ratio,
        report_to="none",
        remove_unused_columns=False,
    )

    lora_config = create_lora_config(tc)

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        peft_config=lora_config,
        max_seq_length=tc.max_seq_length,
        tokenizer=tokenizer,
        args=training_args,
    )

    print(f"  Starting training (iteration {iteration})...")
    print(f"    Dataset size: {len(train_dataset)}")
    print(f"    Max steps: {tc.max_steps} | Epochs: {tc.num_train_epochs}")
    print(f"    LoRA r={tc.lora_r}, alpha={tc.lora_alpha}")

    result = trainer.train()

    print(f"  Training complete. Loss: {result.training_loss:.4f}")
    return trainer, result


def merge_and_save(trainer, tokenizer, output_dir: str):
    """Merge LoRA weights into base model and save."""
    print("  Merging LoRA weights...")
    merged_model = trainer.model.merge_and_unload()

    merged_path = os.path.join(output_dir, "merged_model")
    os.makedirs(merged_path, exist_ok=True)

    merged_model.save_pretrained(merged_path)
    tokenizer.save_pretrained(merged_path)

    print(f"  Merged model saved to: {merged_path}")
    return merged_path


def push_to_hub(model_path: str, repo_id: str, hf_token: str):
    """Push the merged model and tokenizer to HuggingFace Hub."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"  Pushing model to HuggingFace Hub: {repo_id}")

    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    model.push_to_hub(repo_id, token=hf_token)
    tokenizer.push_to_hub(repo_id, token=hf_token)

    print(f"  ✓ Model pushed to: https://huggingface.co/{repo_id}")
    return f"https://huggingface.co/{repo_id}"


def generate_responses(
    model_path: str,
    records: list[dict],
    model_info: ModelInfo,
    hf_token: str,
    max_new_tokens: int = 128,
    batch_size: int = 1,
) -> list[dict]:
    """
    Generate responses for validation records using the fine-tuned model.

    Adds 'FT' field to each record with the model's response.
    """
    torch = _import_torch()
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    import copy

    print(f"  Generating responses for {len(records)} records...")

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
        token=hf_token,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        token=hf_token,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    results = copy.deepcopy(records)

    for i, rec in enumerate(results):
        question = rec.get("Question", "")
        prompt = format_inference_prompt(model_info.template_key, question)

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.5,
                top_k=5,
                repetition_penalty=1.2,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )

        full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract only the generated response (after the prompt)
        prompt_text = tokenizer.decode(
            tokenizer(prompt, return_tensors="pt")["input_ids"][0],
            skip_special_tokens=True,
        )
        response = full_output[len(prompt_text):].strip()

        # Clean up common artifacts
        for stop_token in ["</s>", "<|end|>", "<|im_end|>", "<end_of_turn>",
                           "<|endoftext|>", "### End", "<|end▁of▁sentence|>"]:
            if stop_token in response:
                response = response[:response.index(stop_token)].strip()

        rec["FT"] = response

        if (i + 1) % 20 == 0 or (i + 1) == len(results):
            print(f"    Generated {i+1}/{len(results)}")

    # Clean up GPU memory
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


def cleanup_gpu():
    """Force GPU memory cleanup."""
    gc.collect()
    try:
        torch = _import_torch()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
