"""
Prompt templates for each supported SLM family.

Each template defines how to format a user/assistant conversation
for training and inference. The template must match what the base
model was pre-trained with.
"""

from typing import Callable


def _tinyllama_template(question: str, answer: str | None = None) -> str:
    if answer is not None:
        return f"<|user|>\n{question}</s>\n<|assistant|>\n{answer}</s>"
    return f"<|user|>\n{question}</s>\n<|assistant|>\n"


def _gemma_template(question: str, answer: str | None = None) -> str:
    if answer is not None:
        return f"<start_of_turn>user\n{question}<end_of_turn>\n<start_of_turn>model\n{answer}<end_of_turn>"
    return f"<start_of_turn>user\n{question}<end_of_turn>\n<start_of_turn>model\n"


def _deepseek_template(question: str, answer: str | None = None) -> str:
    if answer is not None:
        return f"<|begin▁of▁sentence|>User: {question}\n\nAssistant: {answer}<|end▁of▁sentence|>"
    return f"<|begin▁of▁sentence|>User: {question}\n\nAssistant: "


def _phi3_template(question: str, answer: str | None = None) -> str:
    if answer is not None:
        return f"<|user|>\n{question}<|end|>\n<|assistant|>\n{answer}<|end|>"
    return f"<|user|>\n{question}<|end|>\n<|assistant|>\n"


def _qwen_template(question: str, answer: str | None = None) -> str:
    if answer is not None:
        return (
            f"<|im_start|>user\n{question}<|im_end|>\n"
            f"<|im_start|>assistant\n{answer}<|im_end|>"
        )
    return f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"


def _minicpm_template(question: str, answer: str | None = None) -> str:
    if answer is not None:
        return f"<用户>{question}<AI>{answer}"
    return f"<用户>{question}<AI>"


def _h2o_danube_template(question: str, answer: str | None = None) -> str:
    if answer is not None:
        return f"<|prompt|>{question}</s><|answer|>{answer}</s>"
    return f"<|prompt|>{question}</s><|answer|>"


def _smollm_template(question: str, answer: str | None = None) -> str:
    # SmolLM uses ChatML-style formatting
    return _chatml_template(question, answer)


def _stablelm_template(question: str, answer: str | None = None) -> str:
    if answer is not None:
        return f"<|user|>\n{question}<|endoftext|>\n<|assistant|>\n{answer}<|endoftext|>"
    return f"<|user|>\n{question}<|endoftext|>\n<|assistant|>\n"


def _mobilellama_template(question: str, answer: str | None = None) -> str:
    # MobileLLaMA uses standard Llama-2 chat format
    if answer is not None:
        return f"[INST] {question} [/INST] {answer}"
    return f"[INST] {question} [/INST] "


def _mobillama_template(question: str, answer: str | None = None) -> str:
    return _mobilellama_template(question, answer)


def _fox_template(question: str, answer: str | None = None) -> str:
    return _chatml_template(question, answer)


def _dolly_template(question: str, answer: str | None = None) -> str:
    if answer is not None:
        return (
            f"### Instruction:\n{question}\n\n"
            f"### Response:\n{answer}\n\n### End"
        )
    return f"### Instruction:\n{question}\n\n### Response:\n"


def _olmo_template(question: str, answer: str | None = None) -> str:
    return _chatml_template(question, answer)


def _chatml_template(question: str, answer: str | None = None) -> str:
    """Generic ChatML template used as fallback."""
    if answer is not None:
        return (
            f"<|im_start|>user\n{question}<|im_end|>\n"
            f"<|im_start|>assistant\n{answer}<|im_end|>"
        )
    return f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"


TEMPLATE_MAP: dict[str, Callable] = {
    "tinyllama": _tinyllama_template,
    "gemma": _gemma_template,
    "deepseek": _deepseek_template,
    "phi3": _phi3_template,
    "qwen": _qwen_template,
    "minicpm": _minicpm_template,
    "h2o_danube": _h2o_danube_template,
    "smollm": _smollm_template,
    "stablelm": _stablelm_template,
    "mobilellama": _mobilellama_template,
    "mobillama": _mobillama_template,
    "fox": _fox_template,
    "dolly": _dolly_template,
    "olmo": _olmo_template,
    "chatml": _chatml_template,
}


def get_template(template_key: str) -> Callable:
    """Get the prompt template function for a given model template key."""
    if template_key not in TEMPLATE_MAP:
        print(f"  ⚠ Unknown template '{template_key}', falling back to chatml")
        return TEMPLATE_MAP["chatml"]
    return TEMPLATE_MAP[template_key]


def format_training_example(template_key: str, question: str, answer: str) -> str:
    """Format a Q&A pair for training using the model's prompt template."""
    template_fn = get_template(template_key)
    return template_fn(question, answer)


def format_inference_prompt(template_key: str, question: str) -> str:
    """Format a question for inference (no answer) using the model's prompt template."""
    template_fn = get_template(template_key)
    return template_fn(question, None)
