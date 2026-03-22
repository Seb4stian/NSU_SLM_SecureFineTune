"""
Model registry: maps friendly model names to HuggingFace repo IDs and metadata.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelInfo:
    repo_id: str
    friendly_name: str
    template_key: str
    default_max_seq_length: int = 512
    supports_bf16: bool = False
    description: str = ""


MODEL_REGISTRY: dict[str, ModelInfo] = {
    # ── TinyLlama ──────────────────────────────────────────────────────────
    "tinyllama": ModelInfo(
        repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        friendly_name="TinyLlama 1.1B Chat",
        template_key="tinyllama",
        default_max_seq_length=2048,
        description="1.1B parameter chat model based on Llama architecture",
    ),
    # ── Gemma ──────────────────────────────────────────────────────────────
    "gemma-2b-it": ModelInfo(
        repo_id="google/gemma-2b-it",
        friendly_name="Gemma 2B IT",
        template_key="gemma",
        default_max_seq_length=2048,
        description="Google's 2B parameter instruction-tuned model",
    ),
    # ── DeepSeek R1 Distill ────────────────────────────────────────────────
    "deepseek-r1-distill-qwen-1.5b": ModelInfo(
        repo_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        friendly_name="DeepSeek R1 Distill Qwen 1.5B",
        template_key="deepseek",
        default_max_seq_length=2048,
        supports_bf16=True,
        description="DeepSeek R1 distilled into Qwen 1.5B",
    ),
    # ── Phi-3 ──────────────────────────────────────────────────────────────
    "phi-3-mini-4k-instruct": ModelInfo(
        repo_id="microsoft/Phi-3-mini-4k-instruct",
        friendly_name="Phi-3 Mini 4K Instruct",
        template_key="phi3",
        default_max_seq_length=4096,
        supports_bf16=True,
        description="Microsoft Phi-3 Mini with 4K context",
    ),
    # ── Qwen ───────────────────────────────────────────────────────────────
    "qwen1.5-0.5b-chat": ModelInfo(
        repo_id="Qwen/Qwen1.5-0.5B-Chat",
        friendly_name="Qwen 1.5 0.5B Chat",
        template_key="qwen",
        default_max_seq_length=2048,
        supports_bf16=True,
        description="Qwen 1.5 0.5B chat model",
    ),
    # ── MiniCPM ────────────────────────────────────────────────────────────
    "minicpm-1b-sft-bf16": ModelInfo(
        repo_id="openbmb/MiniCPM-1B-sft-bf16",
        friendly_name="MiniCPM 1B SFT",
        template_key="minicpm",
        default_max_seq_length=2048,
        supports_bf16=True,
        description="OpenBMB MiniCPM 1B supervised fine-tuned",
    ),
    # ── H2O Danube ─────────────────────────────────────────────────────────
    "h2o-danube-1.8b-sft": ModelInfo(
        repo_id="h2oai/h2o-danube-1.8b-sft",
        friendly_name="H2O Danube 1.8B SFT",
        template_key="h2o_danube",
        default_max_seq_length=2048,
        description="H2O.ai Danube 1.8B supervised fine-tuned",
    ),
    # ── SmolLM ─────────────────────────────────────────────────────────────
    "smollm-135m-instruct": ModelInfo(
        repo_id="HuggingFaceTB/SmolLM-135M-Instruct",
        friendly_name="SmolLM 135M Instruct",
        template_key="smollm",
        default_max_seq_length=2048,
        description="HuggingFace SmolLM 135M instruction-tuned",
    ),
    # ── StableLM ───────────────────────────────────────────────────────────
    "stablelm-2-zephyr-1.6b": ModelInfo(
        repo_id="stabilityai/stablelm-2-zephyr-1_6b",
        friendly_name="StableLM 2 Zephyr 1.6B",
        template_key="stablelm",
        default_max_seq_length=4096,
        description="Stability AI StableLM 2 Zephyr 1.6B",
    ),
    # ── MobileLLaMA ────────────────────────────────────────────────────────
    "mobilellama-1.4b-chat": ModelInfo(
        repo_id="mtgv/MobileLLaMA-1.4B-Chat",
        friendly_name="MobileLLaMA 1.4B Chat",
        template_key="mobilellama",
        default_max_seq_length=2048,
        description="MobileLLaMA 1.4B chat model",
    ),
    # ── MobiLlama ──────────────────────────────────────────────────────────
    "mobillama-0.5b-chat": ModelInfo(
        repo_id="MBZUAI/MobiLlama-05B-Chat",
        friendly_name="MobiLlama 0.5B Chat",
        template_key="mobillama",
        default_max_seq_length=2048,
        description="MBZUAI MobiLlama 0.5B chat model",
    ),
    # ── Fox ─────────────────────────────────────────────────────────────────
    "fox-1-1.6b-instruct": ModelInfo(
        repo_id="tensoropera/Fox-1-1.6B-Instruct-v0.1",
        friendly_name="Fox 1 1.6B Instruct",
        template_key="fox",
        default_max_seq_length=2048,
        description="TensorOpera Fox 1 1.6B instruction-tuned",
    ),
    # ── Dolly ──────────────────────────────────────────────────────────────
    "dolly-v1-6b": ModelInfo(
        repo_id="databricks/dolly-v1-6b",
        friendly_name="Dolly v1 6B",
        template_key="dolly",
        default_max_seq_length=2048,
        description="Databricks Dolly v1 6B",
    ),
    # ── OLMo ───────────────────────────────────────────────────────────────
    "olmo-7b-instruct": ModelInfo(
        repo_id="allenai/OLMo-7B-Instruct-hf",
        friendly_name="OLMo 7B Instruct",
        template_key="olmo",
        default_max_seq_length=2048,
        description="Allen AI OLMo 7B instruction-tuned",
    ),
}


def resolve_model(name: str) -> ModelInfo:
    """
    Resolve a model name to its ModelInfo.
    Accepts friendly names, registry keys, or full HuggingFace repo IDs.
    """
    key = name.lower().strip()

    # Direct registry match
    if key in MODEL_REGISTRY:
        return MODEL_REGISTRY[key]

    # Match by repo_id (case-insensitive)
    for info in MODEL_REGISTRY.values():
        if info.repo_id.lower() == key:
            return info

    # Partial match on friendly name or repo_id
    for k, info in MODEL_REGISTRY.items():
        if key in k or key in info.repo_id.lower() or key in info.friendly_name.lower():
            return info

    # Not found in registry → create a generic entry
    print(f"  ⚠ Model '{name}' not in registry. Using generic chatml template.")
    return ModelInfo(
        repo_id=name,
        friendly_name=name,
        template_key="chatml",
        description="Custom model (not in registry)",
    )


def list_supported_models() -> list[dict]:
    """Return a list of all supported models with their details."""
    return [
        {
            "key": key,
            "name": info.friendly_name,
            "repo_id": info.repo_id,
            "description": info.description,
        }
        for key, info in MODEL_REGISTRY.items()
    ]
