"""
Configuration loader for the Secure Fine-Tune Framework.

Reads a YAML config file containing API keys, model parameters,
dataset paths, and judge settings.
"""

import os
import yaml
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class JudgeConfig:
    provider: str = "openai"  # openai, anthropic, xai
    model: str = "gpt-4o"
    api_key: str = ""
    temperature: float = 0.1
    max_tokens: int = 2048


@dataclass
class TrainingConfig:
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    per_device_train_batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    num_train_epochs: int = 3
    max_steps: int = 250
    fp16: bool = True
    bf16: bool = False
    save_strategy: str = "epoch"
    logging_steps: int = 10
    optim: str = "paged_adamw_32bit"
    max_seq_length: int = 512
    warmup_ratio: float = 0.03


@dataclass
class FrameworkConfig:
    model_name: str = ""
    training_dataset: str = ""
    validation_dataset: str = ""
    output_dir: str = "./output"
    hf_token: str = ""
    hf_repo_id: str = ""
    max_iterations: int = 5
    target_score: float = 0.95
    judge: JudgeConfig = field(default_factory=JudgeConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    refusal_phrase: str = "I am sorry, I do not have an answer for your question"
    task_domain: str = "JavaScript libraries"
    task_description: str = "Answer questions about JavaScript libraries and frameworks"


def load_config(config_path: str) -> FrameworkConfig:
    """Load and validate a YAML configuration file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if raw is None:
        raise ValueError("Config file is empty")

    cfg = FrameworkConfig()

    # Top-level fields
    cfg.model_name = raw.get("model_name", cfg.model_name)
    cfg.training_dataset = raw.get("training_dataset", cfg.training_dataset)
    cfg.validation_dataset = raw.get("validation_dataset", cfg.validation_dataset)
    cfg.output_dir = raw.get("output_dir", cfg.output_dir)
    cfg.hf_token = raw.get("hf_token", os.environ.get("HF_TOKEN", cfg.hf_token))
    cfg.hf_repo_id = raw.get("hf_repo_id", cfg.hf_repo_id)
    cfg.max_iterations = min(raw.get("max_iterations", cfg.max_iterations), 10)
    cfg.target_score = raw.get("target_score", cfg.target_score)
    cfg.refusal_phrase = raw.get("refusal_phrase", cfg.refusal_phrase)
    cfg.task_domain = raw.get("task_domain", cfg.task_domain)
    cfg.task_description = raw.get("task_description", cfg.task_description)

    # Judge config
    judge_raw = raw.get("judge", {})
    cfg.judge.provider = judge_raw.get("provider", cfg.judge.provider)
    cfg.judge.model = judge_raw.get("model", cfg.judge.model)
    cfg.judge.api_key = judge_raw.get("api_key",
                                       os.environ.get("JUDGE_API_KEY", cfg.judge.api_key))
    cfg.judge.temperature = judge_raw.get("temperature", cfg.judge.temperature)
    cfg.judge.max_tokens = judge_raw.get("max_tokens", cfg.judge.max_tokens)

    # Training config
    train_raw = raw.get("training", {})
    for fld in [
        "lora_r", "lora_alpha", "lora_dropout",
        "per_device_train_batch_size", "gradient_accumulation_steps",
        "learning_rate", "lr_scheduler_type", "num_train_epochs",
        "max_steps", "fp16", "bf16", "save_strategy", "logging_steps",
        "optim", "max_seq_length", "warmup_ratio",
    ]:
        if fld in train_raw:
            setattr(cfg.training, fld, train_raw[fld])

    _validate_config(cfg)
    return cfg


def _validate_config(cfg: FrameworkConfig):
    """Validate required fields."""
    errors = []
    if not cfg.model_name:
        errors.append("model_name is required")
    if not cfg.training_dataset:
        errors.append("training_dataset is required")
    if not cfg.validation_dataset:
        errors.append("validation_dataset is required")
    if not cfg.judge.api_key:
        errors.append("judge.api_key is required (or set JUDGE_API_KEY env var)")
    if not cfg.hf_token:
        errors.append("hf_token is required (or set HF_TOKEN env var)")
    if cfg.max_iterations < 1 or cfg.max_iterations > 10:
        errors.append("max_iterations must be between 1 and 10")

    if errors:
        raise ValueError("Configuration errors:\n  - " + "\n  - ".join(errors))
