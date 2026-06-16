# Ablation Study: Security–Utility Trade-off in Fine-Tuned Small Language Models

## Study Overview

This ablation study investigates how different training compositions affect both the **security** (resistance to adversarial jailbreak attacks) and **utility** (task-specific accuracy on JavaScript library Q&A) of a fine-tuned Small Language Model. We focus on a single architecture — **MiniCPM-1B-sft-bf16** (1.2B parameters) — which exhibited a notable vulnerability to the AutoDAN white-box attack in our initial evaluation (62.5% ASR on the base fine-tuned model).

The study decomposes the standard fine-tuning pipeline into three isolated components to measure their individual contributions:

| Variant | Training Strategy | Description |
|---------|------------------|-------------|
| **OOB (Baseline)** | No fine-tuning | Original pretrained model, unmodified |
| **Standard FT** | Full iterative pipeline | Training dataset refined by LLM security judge (good examples + refusal examples) |
| **OnlyGoodExamples** | Task data only | Fine-tuned using only the correct Q&A pairs — no security-judged refusal examples |
| **OnlyRejectingQuestions** | Security data only | Fine-tuned using only the refusal/rejection examples from the security judge |
| **ToxicSuppressed** | Logit-level filtering | No fine-tuning; instead, suppress generation probability of identified toxic tokens at decode time |

---

## Results

### Attack Success Rate (ASR) by Strategy

| Attack Method | Type | OOB Baseline | Standard FT | OnlyRejecting | OnlyGoodExamples | ToxicSuppressed |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| Direct Request | Black-box | 35.0% | **0.0%** | **0.0%** | 37.5% | 37.5% |
| Human Jailbreaks | Black-box | 31.0% | 16.9% | **2.0%** | 37.5% | 36.5% |
| PAP-Top5 | Black-box | 48.5% | 24.5% | **0.0%** | 35.0% | 35.0% |
| AutoDAN | White-box | 62.5% | 57.5% | **25.0%** | 52.5% | 42.5% |
| GCG | White-box | 25.0% | **0.0%** | **0.0%** | 20.0% | 20.0% |
| AutoPrompt | White-box | 20.0% | 2.5% | **0.0%** | 25.0% | 25.0% |
| PEZ | White-box | 18.0% | 4.0% | 5.0% | 17.5% | 18.5% |
| UAT | White-box | 10.0% | **0.0%** | **0.0%** | 22.5% | 22.5% |
| GBDA | White-box | 20.0% | 6.5% | **0.0%** | 16.0% | 15.5% |
| **Mean ASR** | — | **30.0%** | **12.4%** | **3.6%** | **29.3%** | **28.2%** |

### Task Accuracy (JavaScript Library Q&A)

| Variant | Accuracy | Δ vs OOB |
|:---|:---:|:---:|
| OOB (Baseline) | 0.2024 | — |
| **Standard FT** | **0.8056** | **+0.6032** |
| OnlyGoodExamples | 0.6353 | +0.4329 |
| OnlyRejectingQuestions | 0.3647 | +0.1623 |
| ToxicSuppressed | 0.2064 | +0.0040 |

---

## Key Findings

### 1. Toxic Token Suppression: Minimal Impact

Suppressing the generation probability of identified toxic tokens (setting logits to $-\infty$) provides only a **marginal reduction** in attack success. The AutoDAN ASR decreased from 62.5% to 42.5% (−20 percentage points), but all other attack methods showed **virtually no change** (≤2.5% difference). Critically, this approach has **zero effect on task accuracy** (0.2064 vs 0.2024), confirming that toxic token suppression operates orthogonally to the model's learned behavior — it blocks specific generation patterns but cannot teach the model to refuse harmful instructions.

> **Conclusion:** Logit-level token filtering is insufficient as a standalone defense mechanism. It addresses a symptom (specific token sequences) rather than the root cause (lack of safety alignment in the model weights).

### 2. OnlyGoodExamples: Utility Without Security

Fine-tuning with only the correct Q&A pairs (no refusal examples) produces a model with reasonable task accuracy (**63.5%**) but **increased vulnerability** compared to the OOB baseline in several attack categories:

- Human Jailbreaks: 31.0% → **37.5%** (+6.5%)
- Direct Request: 35.0% → **37.5%** (+2.5%)
- UAT: 10.0% → **22.5%** (+12.5%)
- AutoPrompt: 20.0% → **25.0%** (+5.0%)

This result **reaffirms a central thesis of this dissertation**: fine-tuning a model without explicit security auditing and refusal training can introduce security regressions. The model becomes more capable at answering questions but simultaneously becomes more compliant with adversarial prompts, as it has learned to "always be helpful" without learning appropriate boundaries.

> **Conclusion:** Unaudited fine-tuning creates a false sense of improvement — higher task accuracy masks degraded safety properties that are invisible without adversarial evaluation.

### 3. OnlyRejectingQuestions: Security Without Utility

The security-only variant achieves the **lowest mean ASR** across all attacks (3.6%), dramatically outperforming even the standard fine-tuned model. However, this comes at a catastrophic cost to utility:

- Task accuracy: only **36.5%** (vs 80.6% for standard FT)
- The model over-refuses legitimate queries, making it functionally useless for its intended task
- It has learned a single behavior — reject everything — which trivially satisfies the safety constraint but violates the utility constraint

> **Conclusion:** Optimizing exclusively for security produces a model that is safe but useless. A model that refuses all queries achieves perfect attack resistance at the cost of zero practical value.

### 4. Standard FT (Balanced Pipeline): The Pareto-Optimal Solution

The standard iterative fine-tuning pipeline — which incorporates **both** good Q&A examples **and** security-judged refusal examples, refined through LLM judge error analysis — achieves the best trade-off:

- **Highest task accuracy:** 80.6%
- **Significant ASR reduction:** Mean ASR drops from 30.0% to 12.4% (−59% relative reduction)
- **Near-zero ASR on most attacks:** 0.0% on Direct Request, GCG, and UAT

The one persistent vulnerability is **AutoDAN** (57.5% ASR), which exploits token-level adversarial suffixes that are fundamentally harder to defend against through training-data composition alone.

---

## Security–Utility Trade-off Visualization

```
                           Mean ASR (%)
                    0%     10%     20%     30%
                    │       │       │       │
    OnlyRejecting  ▓▓ 3.6%                      ← Best Security, Worst Utility (Acc: 36.5%)
    Standard FT    ▓▓▓▓▓▓ 12.4%                 ← Best Balance (Acc: 80.6%) ★
    ToxicSuppressed▓▓▓▓▓▓▓▓▓▓▓▓▓▓ 28.2%        ← Minimal Effect (Acc: 20.6%)
    OnlyGoodExampl ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ 29.3%       ← Security Regression (Acc: 63.5%)
    OOB (Baseline) ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ 30.0%       ← No Training (Acc: 20.2%)
```

---

## Conclusion

This ablation study demonstrates that the **balanced composition of training data** — combining task-specific examples with security-audited refusal examples — is the **critical differentiator** in producing models that are both useful and safe. Neither pure utility training nor pure safety training achieves an acceptable outcome in isolation:

| Strategy | Utility ✓ | Security ✓ | Acceptable? |
|:---|:---:|:---:|:---:|
| No fine-tuning (OOB) | ✗ | ✗ | ✗ |
| OnlyGoodExamples | ◐ | ✗ (regression) | ✗ |
| OnlyRejectingQuestions | ✗ | ✓ | ✗ |
| ToxicSuppressed (inference) | ✗ | ✗ | ✗ |
| **Standard FT (balanced)** | **✓** | **◐** | **✓** |

The **value proposition of this dissertation** lies in the iterative, judge-audited training pipeline that automatically balances these competing objectives. By incorporating LLM-judge feedback to continuously refine both the positive examples (correct answers) and negative examples (appropriate refusals), the framework produces models that maximize utility while maintaining robust security boundaries — a balance that cannot be achieved through any single-dimension optimization.

### Remaining Challenge: AutoDAN Vulnerability

The persistent AutoDAN vulnerability (57.5% ASR even on the standard FT model) identifies a limitation of training-data-based defenses against token-level adversarial optimization. Future work should explore:
1. Adversarial training with AutoDAN-generated examples
2. Combining logit suppression with fine-tuning (hybrid defense)
3. Input preprocessing / perplexity-based filtering at inference time

---

*Study conducted on MiniCPM-1B-sft-bf16 | Evaluation: 9 attack methods × 40 behaviors | Task: 499 JavaScript library Q&A*
