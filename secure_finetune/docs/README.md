# Secure Fine-Tune Framework

**Iterative Small Language Model Fine-Tuning with LLM Judge Feedback**

An automated framework for fine-tuning small language models (SLMs) from HuggingFace using LoRA/PEFT, with an external LLM judge (OpenAI, Anthropic, or xAI) that evaluates model outputs, performs error analysis, and iteratively improves the training dataset — up to 10 iterations — until the model meets a target quality score.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Installation](#installation)
3. [Configuration File](#configuration-file)
4. [Supported Models](#supported-models)
5. [How It Works](#how-it-works)
6. [Running the Framework](#running-the-framework)
7. [Understanding the Results](#understanding-the-results)
8. [Dataset Format](#dataset-format)
9. [GPT-Judge Providers](#gpt-judge-providers)
10. [Advanced Configuration](#advanced-configuration)
11. [Troubleshooting](#troubleshooting)

---

## Quick Start

```bash
# 1. Install dependencies
pip install -e .

# 2. Copy and edit the config template
cp secure_finetune/config_template.yaml my_config.yaml
# Edit my_config.yaml with your API keys, dataset paths, and model choice

# 3. Validate your config (dry run)
python -m secure_finetune my_config.yaml --dry-run

# 4. Run the framework
python -m secure_finetune my_config.yaml
```

---

## Installation

### Prerequisites

- Python 3.10 or higher
- CUDA-compatible GPU (recommended, 8GB+ VRAM)
- A HuggingFace account and access token
- An API key for at least one judge provider (OpenAI, Anthropic, or xAI)

### Install

From the repository root:

```bash
pip install -e .
```

This installs all required dependencies:

| Package         | Purpose                              |
|-----------------|--------------------------------------|
| `torch`         | Deep learning framework              |
| `transformers`  | HuggingFace model loading/training   |
| `peft`          | LoRA adapter fine-tuning             |
| `trl`           | SFTTrainer for supervised fine-tuning|
| `bitsandbytes`  | 4-bit quantization                   |
| `accelerate`    | Multi-GPU / mixed precision          |
| `datasets`      | HuggingFace dataset utilities        |
| `scikit-learn`  | ML metrics computation               |
| `openai`        | OpenAI / xAI judge API               |
| `anthropic`     | Anthropic judge API                  |
| `pyyaml`        | Configuration file parsing           |

---

## Configuration File

The framework is driven by a single YAML configuration file. Copy the template and fill in your values:

```bash
cp secure_finetune/config_template.yaml my_config.yaml
```

### Required Fields

```yaml
# Which model to fine-tune (see Supported Models below)
model_name: "tinyllama"

# Paths to your JSONL datasets
training_dataset: "C:\\path\\to\\training.jsonl"
validation_dataset: "C:\\path\\to\\validation.jsonl"

# HuggingFace credentials
hf_token: "hf_YOUR_TOKEN_HERE"
hf_repo_id: "YourUsername/your-model-name"

# Judge LLM credentials
judge:
  provider: "openai"          # openai | anthropic | xai
  model: "gpt-4o"
  api_key: "sk-YOUR_KEY_HERE"
```

### Optional Fields

```yaml
# Where to save checkpoints and results (default: ./output)
output_dir: "./output_finetune"

# Stop after this many iterations (1-10, hard cap at 10)
max_iterations: 5

# Stop early if average score reaches this threshold
target_score: 0.95

# What domain the model should answer about
task_domain: "JavaScript libraries"
task_description: "Answer questions about JavaScript libraries and frameworks"

# The phrase the model should use when refusing out-of-scope questions
refusal_phrase: "I am sorry, I do not have an answer for your question"
```

### Training Hyperparameters

```yaml
training:
  lora_r: 8                          # LoRA rank
  lora_alpha: 16                     # LoRA alpha scaling
  lora_dropout: 0.05                 # LoRA dropout
  per_device_train_batch_size: 8     # Batch size per GPU
  gradient_accumulation_steps: 4     # Effective batch = 8 × 4 = 32
  learning_rate: 0.0002              # Learning rate
  lr_scheduler_type: "cosine"        # LR schedule
  num_train_epochs: 3                # Number of epochs
  max_steps: 250                     # Max training steps (-1 for epochs only)
  fp16: true                         # Use FP16 mixed precision
  bf16: false                        # Use BF16 (for Ampere+ GPUs)
  max_seq_length: 512                # Max input sequence length
  warmup_ratio: 0.03                 # Warmup proportion
  optim: "paged_adamw_32bit"         # Optimizer
```

### Judge Provider Examples

**OpenAI:**
```yaml
judge:
  provider: "openai"
  model: "gpt-4o"
  api_key: "sk-YOUR_OPENAI_KEY"
```

**Anthropic:**
```yaml
judge:
  provider: "anthropic"
  model: "claude-sonnet-4-20250514"
  api_key: "sk-ant-YOUR_ANTHROPIC_KEY"
```

**xAI (Grok):**
```yaml
judge:
  provider: "xai"
  model: "grok-3"
  api_key: "xai-YOUR_GROK_KEY"
```

---

## Supported Models

Run `python -m secure_finetune --list-models` to see all supported models:

| Key                            | Model Name                    | HuggingFace Repo ID                            |
|--------------------------------|-------------------------------|-------------------------------------------------|
| `tinyllama`                    | TinyLlama 1.1B Chat          | `TinyLlama/TinyLlama-1.1B-Chat-v1.0`           |
| `gemma-2b-it`                  | Gemma 2B IT                  | `google/gemma-2b-it`                            |
| `deepseek-r1-distill-qwen-1.5b`| DeepSeek R1 Distill Qwen 1.5B| `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`    |
| `phi-3-mini-4k-instruct`      | Phi-3 Mini 4K Instruct       | `microsoft/Phi-3-mini-4k-instruct`              |
| `qwen1.5-0.5b-chat`           | Qwen 1.5 0.5B Chat           | `Qwen/Qwen1.5-0.5B-Chat`                       |
| `minicpm-1b-sft-bf16`         | MiniCPM 1B SFT               | `openbmb/MiniCPM-1B-sft-bf16`                  |
| `h2o-danube-1.8b-sft`         | H2O Danube 1.8B SFT          | `h2oai/h2o-danube-1.8b-sft`                    |
| `smollm-135m-instruct`        | SmolLM 135M Instruct         | `HuggingFaceTB/SmolLM-135M-Instruct`           |
| `stablelm-2-zephyr-1.6b`      | StableLM 2 Zephyr 1.6B       | `stabilityai/stablelm-2-zephyr-1_6b`           |
| `mobilellama-1.4b-chat`       | MobileLLaMA 1.4B Chat        | `mtgv/MobileLLaMA-1.4B-Chat`                   |
| `mobillama-0.5b-chat`         | MobiLlama 0.5B Chat          | `MBZUAI/MobiLlama-05B-Chat`                    |
| `fox-1-1.6b-instruct`         | Fox 1 1.6B Instruct          | `tensoropera/Fox-1-1.6B-Instruct-v0.1`         |
| `dolly-v1-6b`                 | Dolly v1 6B                  | `databricks/dolly-v1-6b`                       |
| `olmo-7b-instruct`            | OLMo 7B Instruct             | `allenai/OLMo-7B-Instruct-hf`                  |

You can also use any HuggingFace model repo ID not in this list — the framework will use a generic ChatML template as fallback.

---

## How It Works

The framework runs an iterative fine-tuning loop:

```
┌──────────────────────────────────────────────────────────┐
│                  ITERATION LOOP (max 10)                 │
│                                                          │
│  1. Load base model + training dataset                   │
│  2. Fine-tune with LoRA/PEFT (SFTTrainer)                │
│  3. Merge LoRA weights → save merged model               │
│  4. Generate responses on validation dataset              │
│  5. Score each response with GPT-judge (0 or 1)          │
│  6. Compute ML metrics (accuracy, F1, MCC, etc.)         │
│  7. If score < target AND iterations remain:             │
│     a. GPT-judge analyzes errors on 3 axes:              │
│        - Formatting: Is output properly formatted?       │
│        - Topic: Is it answering the right domain?        │
│        - Security: Does it refuse jailbreaks?            │
│     b. Judge proposes dataset modifications:             │
│        - Additions (new Q&A pairs for weak areas)        │
│        - Modifications (fix incorrect entries)           │
│        - Removals (delete harmful/duplicate entries)     │
│     c. Apply modifications → go to step 1               │
│  8. Push best model to HuggingFace Hub                   │
│  9. Save final dataset, metrics, and run summary         │
└──────────────────────────────────────────────────────────┘
```

### GPT-Judge Error Analysis Axes

The judge evaluates errors on three critical axes:

1. **Formatting** — Are model outputs concise and properly formatted? (e.g., short library names instead of verbose explanations)
2. **Task Specification** — Is the model answering within the correct topic domain? (e.g., JavaScript libraries, not cooking recipes)
3. **Security** — Does the model properly refuse off-topic, jailbreak, or prompt injection attempts with the standard refusal phrase?

---

## Running the Framework

### List Supported Models

```bash
python -m secure_finetune --list-models
```

### Validate Configuration (Dry Run)

```bash
python -m secure_finetune my_config.yaml --dry-run
```

This loads the config, resolves the model, loads both datasets, prints statistics, and exits without training. Use this to verify everything is correct before a long training run.

### Run Full Training

```bash
python -m secure_finetune my_config.yaml
```

The console will display real-time progress for each iteration:

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ITERATION 1/5
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📋 Step 1: Preparing dataset (3345 records)...
🔧 Step 2: Fine-tuning TinyLlama 1.1B Chat...
💾 Step 3: Merging LoRA weights and saving model...
🔮 Step 4: Generating responses on validation set...
⚖️  Step 5: Scoring responses with openai/gpt-4o...
📊 Step 6: Computing ML metrics...
🔍 Step 7: Error analysis — 102 errors, 437 correct

  ★ New best score: 0.8109 (iteration 1)
```

---

## Understanding the Results

### Output Directory Structure

After a run completes, the output directory contains:

```
output_finetune/
├── run_summary.json                    # Overall run summary
├── training_dataset_iter0.jsonl        # Original training dataset
├── training_dataset_iter1.jsonl        # Modified dataset after iteration 1
├── training_dataset_final.jsonl        # Final version of training dataset
│
├── iter_1/
│   ├── merged_model/                   # Merged fine-tuned model files
│   ├── validation_scored_iter1.jsonl   # Validation results with scores
│   ├── metrics_iter1.json              # ML metrics for this iteration
│   └── error_analysis_iter1.json       # Judge error analysis
│
├── iter_2/
│   ├── merged_model/
│   ├── validation_scored_iter2.jsonl
│   ├── metrics_iter2.json
│   └── error_analysis_iter2.json
│
└── checkpoints_iter1/                  # Training checkpoints (LoRA)
    └── ...
```

### run_summary.json

The final summary includes:

```json
{
  "model_name": "TinyLlama 1.1B Chat",
  "model_repo_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
  "hf_url": "https://huggingface.co/YourUser/your-model",
  "best_iteration": 3,
  "best_score": 0.9120,
  "total_iterations": 3,
  "total_time_minutes": 45.2,
  "final_dataset_size": 3412,
  "initial_dataset_size": 3345,
  "all_metrics": [
    { "iteration": 1, "accuracy": 0.81, "f1_score": 0.85, ... },
    { "iteration": 2, "accuracy": 0.88, "f1_score": 0.91, ... },
    { "iteration": 3, "accuracy": 0.91, "f1_score": 0.93, ... }
  ]
}
```

### metrics_iterN.json

Each iteration produces detailed ML metrics:

```json
{
  "total_records": 499,
  "real_answers": 442,
  "refusals": 57,
  "accuracy": 0.8096,
  "precision": 0.8521,
  "recall": 0.9276,
  "f1_score": 0.8883,
  "mcc": 0.4012,
  "roc_auc": 0.6823,
  "avg_score": 0.8096,
  "confusion_matrix": { "tn": 21, "fp": 36, "fn": 32, "tp": 410 },
  "training_loss": 1.244,
  "dataset_size": 3345
}
```

### Score Progression

At the end of a run, the framework prints a visual score progression:

```
  Score Progression:
    Iter 1: 0.8096 |████████████████████████████████
    Iter 2: 0.8721 |██████████████████████████████████
    Iter 3: 0.9120 |████████████████████████████████████
```

### Validation Scored Files

Each `validation_scored_iterN.jsonl` contains every validation record with the model's response and judge score:

```json
{"Question": "Which library is used for ...", "Answer": "React", "FT": "React", "FT_Score": 1}
{"Question": "How do I cook pasta?", "Answer": "I am sorry, ...", "FT": "I am sorry, ...", "FT_Score": 1}
{"Question": "Which library handles ...", "Answer": "Lodash", "FT": "jQuery", "FT_Score": 0}
```

### Error Analysis Files

Each `error_analysis_iterN.json` shows what the judge found and proposed:

```json
{
  "analysis": {
    "formatting_issues": "Some answers include verbose explanations instead of library names",
    "topic_issues": "Model occasionally answers general programming questions",
    "security_issues": "Model fails to refuse 5 jailbreak-style prompts",
    "summary": "Main weakness is in security refusals and overly verbose formatting"
  },
  "additions": [
    {"Question": "What is the meaning of life?", "Answer": "I am sorry, I do not have an answer for your question"},
    {"Question": "Which JS library handles date formatting?", "Answer": "Moment.js"}
  ],
  "modifications": [
    {"index": 42, "Answer": "Express.js"}
  ],
  "removals": [101, 205]
}
```

---

## Dataset Format

Both training and validation datasets use JSONL format (one JSON object per line):

```jsonl
{"Question": "Which JavaScript library is used for building UIs?", "Answer": "React"}
{"Question": "How do I cook pasta?", "Answer": "I am sorry, I do not have an answer for your question"}
```

### Fields

| Field      | Type   | Description                                                  |
|------------|--------|--------------------------------------------------------------|
| `Question` | string | The user's question                                          |
| `Answer`   | string | Expected answer (a real answer or the refusal phrase)        |

### Refusal Entries

For out-of-scope questions, the answer must contain the configured refusal phrase:

```json
{"Question": "What is quantum computing?", "Answer": "I am sorry, I do not have an answer for your question"}
```

---

## GPT-Judge Providers

The framework supports three LLM providers for the judge:

| Provider    | Config key   | Example models                        | API key prefix |
|-------------|-------------|---------------------------------------|----------------|
| **OpenAI**  | `openai`    | `gpt-4o`, `gpt-4o-mini`, `gpt-5.1`   | `sk-`          |
| **Anthropic**| `anthropic`| `claude-sonnet-4-20250514`, `claude-3-haiku-20240307` | `sk-ant-`      |
| **xAI**     | `xai`       | `grok-3`, `grok-2`                    | `xai-`         |

API keys can be set in the config file or via environment variables (`JUDGE_API_KEY` for judge, `HF_TOKEN` for HuggingFace).

---

## Advanced Configuration

### Using a Custom Model (Not in Registry)

Set `model_name` to the full HuggingFace repo ID:

```yaml
model_name: "some-org/custom-model-1b"
```

The framework will use a generic ChatML prompt template as fallback.

### Adjusting the Iteration Loop

```yaml
max_iterations: 10    # Maximum allowed
target_score: 0.98    # Very high quality bar
```

The loop stops when **either** the target score is reached **or** max iterations is hit.

### Using the Standalone Metrics Evaluator

You can run `evaluate_metrics.py` directly on any scored JSONL file:

```bash
python -m secure_finetune.evaluate_metrics path/to/scored_validation.jsonl
```

---

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: No module named 'torch'` | Run `pip install -e .` from the repo root |
| `CUDA out of memory` | Reduce `per_device_train_batch_size` or `max_seq_length` |
| `Configuration errors: judge.api_key is required` | Add your API key to the config or set `JUDGE_API_KEY` env var |
| `Model not found on HuggingFace` | Check `model_name` spelling; some models require `hf_token` for gated access |
| `Target score never reached` | Increase `max_iterations`, adjust training hyperparameters, or review dataset quality |

### GPU Memory Tips

- Use `fp16: true` for most GPUs
- Use `bf16: true` only on Ampere+ GPUs (RTX 3090, A100, etc.)
- Reduce `per_device_train_batch_size` to 4 or 2 for smaller GPUs
- Reduce `max_seq_length` to 256 if sequences are short
