# API Reference

Complete reference for all modules, classes, and functions in the Secure Fine-Tune Framework.

---

## Table of Contents

- [config](#config)
- [model_registry](#model_registry)
- [prompt_templates](#prompt_templates)
- [dataset_manager](#dataset_manager)
- [fine_tuner](#fine_tuner)
- [judge](#judge)
- [evaluator](#evaluator)
- [evaluate_metrics](#evaluate_metrics)
- [main](#main)

---

## config

**Module:** `secure_finetune.config`

Loads and validates a YAML configuration file.

### Dataclasses

#### `JudgeConfig`

Configuration for the GPT-judge LLM provider.

| Field         | Type    | Default    | Description                            |
|---------------|---------|------------|----------------------------------------|
| `provider`    | `str`   | `"openai"` | Judge provider: `openai`, `anthropic`, `xai` |
| `model`       | `str`   | `"gpt-4o"` | Specific model name                    |
| `api_key`     | `str`   | `""`       | API key for the provider               |
| `temperature` | `float` | `0.1`      | Sampling temperature                   |
| `max_tokens`  | `int`   | `2048`     | Maximum tokens in judge response       |

#### `TrainingConfig`

Hyperparameters for LoRA fine-tuning.

| Field                          | Type    | Default              | Description                     |
|--------------------------------|---------|----------------------|---------------------------------|
| `lora_r`                       | `int`   | `8`                  | LoRA rank                       |
| `lora_alpha`                   | `int`   | `16`                 | LoRA alpha scaling factor       |
| `lora_dropout`                 | `float` | `0.05`               | LoRA dropout rate               |
| `per_device_train_batch_size`  | `int`   | `8`                  | Batch size per GPU              |
| `gradient_accumulation_steps`  | `int`   | `4`                  | Gradient accumulation steps     |
| `learning_rate`                | `float` | `2e-4`               | Learning rate                   |
| `lr_scheduler_type`            | `str`   | `"cosine"`           | LR scheduler type               |
| `num_train_epochs`             | `int`   | `3`                  | Number of training epochs       |
| `max_steps`                    | `int`   | `250`                | Max training steps              |
| `fp16`                         | `bool`  | `True`               | Use FP16 mixed precision        |
| `bf16`                         | `bool`  | `False`              | Use BF16 mixed precision        |
| `save_strategy`                | `str`   | `"epoch"`            | When to save checkpoints        |
| `logging_steps`                | `int`   | `10`                 | Log every N steps               |
| `optim`                        | `str`   | `"paged_adamw_32bit"`| Optimizer                       |
| `max_seq_length`               | `int`   | `512`                | Maximum sequence length         |
| `warmup_ratio`                 | `float` | `0.03`               | Warmup ratio                    |

#### `FrameworkConfig`

Top-level configuration container.

| Field                | Type             | Default    | Description                        |
|----------------------|------------------|------------|------------------------------------|
| `model_name`         | `str`            | `""`       | Model key or HuggingFace repo ID   |
| `training_dataset`   | `str`            | `""`       | Path to training JSONL file        |
| `validation_dataset` | `str`            | `""`       | Path to validation JSONL file      |
| `output_dir`         | `str`            | `"./output"` | Output directory                 |
| `hf_token`           | `str`            | `""`       | HuggingFace access token           |
| `hf_repo_id`         | `str`            | `""`       | Target HuggingFace repo ID        |
| `max_iterations`     | `int`            | `5`        | Max fine-tune iterations (cap: 10) |
| `target_score`       | `float`          | `0.95`     | Early-stop score threshold         |
| `judge`              | `JudgeConfig`    | *(defaults)* | Judge configuration              |
| `training`           | `TrainingConfig` | *(defaults)* | Training hyperparameters         |
| `refusal_phrase`     | `str`            | *"I am sorry..."* | Standard refusal phrase     |
| `task_domain`        | `str`            | `"JavaScript libraries"` | Task domain name       |
| `task_description`   | `str`            | *"Answer questions..."* | Task description        |

### Functions

#### `load_config(config_path: str) -> FrameworkConfig`

Load and validate a YAML config file. Raises `FileNotFoundError` if the file doesn't exist, `ValueError` if required fields are missing.

**Parameters:**
- `config_path` — Path to the YAML file

**Returns:** A validated `FrameworkConfig` instance

---

## model_registry

**Module:** `secure_finetune.model_registry`

Maps friendly model names to HuggingFace repository IDs and metadata.

### Dataclass

#### `ModelInfo`

| Field                    | Type   | Default | Description                          |
|--------------------------|--------|---------|--------------------------------------|
| `repo_id`                | `str`  | —       | HuggingFace repo ID                  |
| `friendly_name`          | `str`  | —       | Human-readable model name            |
| `template_key`           | `str`  | —       | Key into `TEMPLATE_MAP`              |
| `default_max_seq_length` | `int`  | `512`   | Default max sequence length          |
| `supports_bf16`          | `bool` | `False` | Whether model supports BF16          |
| `description`            | `str`  | `""`    | Short description                    |

### Constants

#### `MODEL_REGISTRY: dict[str, ModelInfo]`

Dictionary of all 14 supported models keyed by their friendly name.

### Functions

#### `resolve_model(name: str) -> ModelInfo`

Resolve a model name (key, repo ID, or partial match) to its `ModelInfo`. Falls back to a generic ChatML entry for unknown models.

#### `list_supported_models() -> list[dict]`

Returns a list of dicts with `key`, `name`, `repo_id`, and `description` for all registered models.

---

## prompt_templates

**Module:** `secure_finetune.prompt_templates`

Model-specific chat prompt templates for training and inference.

### Template Formats

| Template Key   | Model Family           | Format Example                                          |
|----------------|------------------------|---------------------------------------------------------|
| `tinyllama`    | TinyLlama              | `<\|user\|>\n{q}</s>\n<\|assistant\|>\n{a}</s>`         |
| `gemma`        | Gemma                  | `<start_of_turn>user\n{q}<end_of_turn>\n<start_of_turn>model\n{a}<end_of_turn>` |
| `deepseek`     | DeepSeek R1 Distill    | `<\|begin▁of▁sentence\|>User: {q}\n\nAssistant: {a}<\|end▁of▁sentence\|>` |
| `phi3`         | Phi-3                  | `<\|user\|>\n{q}<\|end\|>\n<\|assistant\|>\n{a}<\|end\|>` |
| `qwen`         | Qwen                   | `<\|im_start\|>user\n{q}<\|im_end\|>\n<\|im_start\|>assistant\n{a}<\|im_end\|>` |
| `minicpm`      | MiniCPM                | `<用户>{q}<AI>{a}`                                      |
| `h2o_danube`   | H2O Danube             | `<\|prompt\|>{q}</s><\|answer\|>{a}</s>`                |
| `smollm`       | SmolLM                 | ChatML format                                           |
| `stablelm`     | StableLM               | `<\|user\|>\n{q}<\|endoftext\|>\n<\|assistant\|>\n{a}<\|endoftext\|>` |
| `mobilellama`  | MobileLLaMA            | `[INST] {q} [/INST] {a}`                               |
| `mobillama`    | MobiLlama              | Same as mobilellama                                     |
| `fox`          | Fox                    | ChatML format                                           |
| `dolly`        | Dolly                  | `### Instruction:\n{q}\n\n### Response:\n{a}\n\n### End` |
| `olmo`         | OLMo                   | ChatML format                                           |
| `chatml`       | Generic fallback       | ChatML format                                           |

### Functions

#### `get_template(template_key: str) -> Callable`

Returns the template function for a given key. Falls back to `chatml` if unknown.

#### `format_training_example(template_key: str, question: str, answer: str) -> str`

Format a Q&A pair for training (includes both question and answer).

#### `format_inference_prompt(template_key: str, question: str) -> str`

Format a question for inference (question only, model generates the answer).

---

## dataset_manager

**Module:** `secure_finetune.dataset_manager`

### Class: `DatasetManager`

```python
DatasetManager(refusal_phrase: str = "I am sorry, ...")
```

#### Methods

| Method | Parameters | Returns | Description |
|--------|-----------|---------|-------------|
| `load_jsonl(filepath)` | `filepath: str` | `list[dict]` | Load a JSONL file into records |
| `save_jsonl(records, filepath)` | `records: list[dict]`, `filepath: str` | `None` | Save records to JSONL |
| `get_dataset_stats(records)` | `records: list[dict]` | `dict` | Returns `total`, `real_answers`, `refusals`, `refusal_ratio` |
| `apply_modifications(records, additions, modifications, removals)` | See below | `list[dict]` | Apply judge-proposed changes |
| `create_versioned_path(original_path, iteration)` | `original_path: str`, `iteration: int` | `str` | Create `{base}_iter{N}.jsonl` path |
| `sample_records(records, n, balanced)` | `records: list[dict]`, `n: int=50`, `balanced: bool=True` | `list[dict]` | Sample records for judge analysis |
| `merge_error_samples(error_records, existing_records)` | `error_records`, `existing_records` | `list[dict]` | Merge without duplicates |

##### `apply_modifications` Parameters

- `records: list[dict]` — Current dataset
- `additions: list[dict]` — New `{"Question": ..., "Answer": ...}` entries to add
- `modifications: list[dict]` — Entries to fix: `{"index": int, "Question": ..., "Answer": ...}`
- `removals: list[int]` — Indices of records to remove

---

## fine_tuner

**Module:** `secure_finetune.fine_tuner`

Handles LoRA/PEFT fine-tuning, model merging, inference, and HuggingFace Hub upload.

### Functions

#### `load_base_model(model_info, config) -> (model, tokenizer)`

Loads the base model with 4-bit quantization (NF4) and returns the model and tokenizer.

#### `prepare_dataset(records, model_info, tokenizer, max_seq_length) -> Dataset`

Converts Q&A records into a HuggingFace `Dataset` with formatted text using the model's prompt template.

#### `create_lora_config(training_config) -> LoraConfig`

Creates a PEFT `LoraConfig` from the training configuration.

#### `train_model(model, tokenizer, train_dataset, config, model_info, iteration) -> (trainer, result)`

Fine-tunes the model using `SFTTrainer` with LoRA. Returns the trainer object and training result.

#### `merge_and_save(trainer, tokenizer, output_dir) -> str`

Merges LoRA weights into the base model and saves to disk. Returns the path to the merged model.

#### `push_to_hub(model_path, repo_id, hf_token) -> str`

Pushes the merged model and tokenizer to HuggingFace Hub. Returns the URL.

#### `generate_responses(model_path, records, model_info, hf_token, max_new_tokens, batch_size) -> list[dict]`

Generates responses for all records using the fine-tuned model. Adds an `"FT"` field to each record.

**Generation config:** `temperature=0.5`, `top_k=5`, `repetition_penalty=1.2`

#### `cleanup_gpu()`

Forces GPU memory cleanup with `gc.collect()` and `torch.cuda.empty_cache()`.

---

## judge

**Module:** `secure_finetune.judge`

Multi-provider GPT-judge for scoring, error analysis, and dataset modification proposals.

### Supported Providers

| Provider Key | Function       | API Base                  |
|-------------|----------------|---------------------------|
| `openai`    | `_call_openai`  | `https://api.openai.com`  |
| `anthropic` | `_call_anthropic`| Anthropic SDK            |
| `xai`/`grok`| `_call_xai`    | `https://api.x.ai/v1`    |

### Functions

#### `score_single(config, question, expected_answer, model_output, task_domain) -> int`

Score a single model output against the expected answer. Returns `0` (incorrect) or `1` (correct).

**Scoring rules:**
- `1` if model output contains at least one correct library name matching expected answer
- `1` if expected answer is a refusal and model also refuses
- `0` if output is irrelevant, wrong, or model answers when it should refuse
- `0` if model refuses when it should provide a real answer

#### `score_batch(config, records, model_output_field, task_domain, delay) -> list[dict]`

Score a batch of records. Adds `"{model_output_field}_Score"` field to each record. Prints progress every 20 records.

**Parameters:**
- `delay: float = 0.2` — Delay between API calls to avoid rate limiting

#### `analyze_errors(config, error_records, correct_records, task_domain, task_description, refusal_phrase, training_sample) -> dict`

Perform error analysis on model mistakes. The judge evaluates errors on three axes (formatting, topic, security) and proposes dataset modifications.

**Returns:**
```python
{
    "analysis": {
        "formatting_issues": str,
        "topic_issues": str,
        "security_issues": str,
        "summary": str,
    },
    "additions": [{"Question": str, "Answer": str}, ...],   # max 50
    "modifications": [{"index": int, "Question": str, "Answer": str}, ...],
    "removals": [int, ...],
}
```

---

## evaluator

**Module:** `secure_finetune.evaluator`

### Class: `Evaluator`

```python
Evaluator(refusal_phrase: str = "I am sorry, ...")
```

#### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `is_refusal(text)` | `bool` | Check if text contains refusal phrase |
| `build_labels_from_scores(records, score_field)` | `(y_true, y_pred)` | Build labels from pre-scored records |
| `build_labels_from_text(records, pred_field)` | `(y_true, y_pred)` | Build labels from response text matching |
| `compute_metrics(records, score_field, pred_field)` | `dict` | Compute all ML metrics |
| `print_metrics(metrics, label)` | `None` | Print formatted metrics table |
| `get_error_records(records, score_field)` | `(errors, correct)` | Split into error/correct lists |
| `save_metrics(metrics, filepath)` | `None` | Save metrics to JSON file |

#### Metrics Computed

| Metric | Key | Description |
|--------|-----|-------------|
| Accuracy | `accuracy` | Overall correctness |
| Precision | `precision` | Positive predictive value |
| Recall | `recall` | True positive rate |
| F1 Score | `f1_score` | Harmonic mean of precision and recall |
| MCC | `mcc` | Matthews Correlation Coefficient |
| ROC-AUC | `roc_auc` | Area under ROC curve |
| Confusion Matrix | `confusion_matrix` | `{tn, fp, fn, tp}` |
| Average Score | `avg_score` | Mean judge score |

---

## evaluate_metrics

**Module:** `secure_finetune.evaluate_metrics`

Standalone ML metrics evaluator (can also be run directly).

```bash
python -m secure_finetune.evaluate_metrics [path_to_scored.jsonl]
```

Supports two modes:
- **Pre-scored mode:** Uses `FT_Score`/`OOB_Score` columns
- **Text-match mode:** Detects refusal phrase in `FT`/`OOB` text columns

---

## main

**Module:** `secure_finetune.main`

CLI entry point and orchestrator.

### `main()`

Parses CLI arguments and runs the iterative fine-tuning loop.

### `run_iteration(...) -> dict`

Executes a single iteration (fine-tune → generate → score → evaluate → analyze).

**Returns:**
```python
{
    "metrics": dict,           # ML metrics
    "scored_records": list,    # Validation records with scores
    "error_analysis": dict,    # Judge analysis and proposals
    "model_path": str,         # Path to merged model
    "errors_count": int,
    "correct_count": int,
}
```

### CLI Arguments

| Argument | Description |
|----------|-------------|
| `config` | Path to YAML configuration file |
| `--list-models` | List all supported models and exit |
| `--dry-run` | Validate config and load datasets without training |
