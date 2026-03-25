# Architecture Overview

Technical documentation of the Secure Fine-Tune Framework's internal architecture, data flow, and module interactions.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         main.py (CLI)                           │
│                    Orchestrates the loop                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────┐  ┌──────────────┐  ┌───────┐  ┌──────────────┐  │
│  │ config.py│  │dataset_manager│  │judge.py│  │ evaluator.py │  │
│  │          │  │     .py       │  │        │  │              │  │
│  │ YAML     │  │ JSONL I/O    │  │ LLM API│  │ ML Metrics   │  │
│  │ loading  │  │ Modify data  │  │ Score  │  │ Accuracy/F1  │  │
│  └────┬─────┘  └──────┬───────┘  │ Analyze│  │ Confusion    │  │
│       │               │          └───┬────┘  └──────┬───────┘  │
│       │               │             │               │          │
│  ┌────┴────────────────┴─────────────┴───────────────┴───────┐  │
│  │                     fine_tuner.py                          │  │
│  │  Load model → LoRA config → SFTTrainer → Merge → Push HF │  │
│  └──────────────────────┬────────────────────────────────────┘  │
│                         │                                       │
│  ┌──────────────────────┴────────────────────────────────────┐  │
│  │              model_registry.py + prompt_templates.py       │  │
│  │          Model lookup       Prompt formatting              │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Data Flow Per Iteration

```
Training JSONL ──┐
                 ├──→ fine_tuner.prepare_dataset() ──→ SFTTrainer.train()
Model Registry ──┘         │
                           ▼
                    fine_tuner.merge_and_save()
                           │
                           ▼
Validation JSONL ──→ fine_tuner.generate_responses() ──→ Records + "FT" field
                                                           │
                                                           ▼
                                                   judge.score_batch()
                                                           │
                                                           ▼
                                              Records + "FT_Score" field
                                                           │
                                          ┌────────────────┴────────────────┐
                                          ▼                                 ▼
                                 evaluator.compute_metrics()     evaluator.get_error_records()
                                          │                                 │
                                          ▼                                 ▼
                                   metrics_iterN.json              judge.analyze_errors()
                                                                           │
                                                                           ▼
                                                              dataset_manager.apply_modifications()
                                                                           │
                                                                           ▼
                                                                  Updated Training JSONL
                                                                    (next iteration)
```

---

## Module Dependency Graph

```
config.py ◄──────────────────── main.py
                                   │
model_registry.py ◄────────────────┤
                                   │
prompt_templates.py ◄──── fine_tuner.py ◄──── main.py
                                   │
dataset_manager.py ◄───────────────┤
                                   │
judge.py ◄─────────────────────────┤
   └── config.py (JudgeConfig)     │
                                   │
evaluator.py ◄─────────────────────┘
```

---

## Mathematical Formalization of the Secure Fine-Tuning Optimization

This section provides a rigorous mathematical description of the entire optimization process, covering the LoRA parameter-efficient fine-tuning, the security-constrained objective, the iterative judge-in-the-loop dataset refinement, and the convergence criteria.

### 1. Notation and Definitions

Let:

$$\mathcal{M}_\theta : \mathcal{X} \rightarrow \mathcal{Y}$$

denote a pre-trained small language model with frozen parameters $\theta \in \mathbb{R}^d$, mapping input sequences $x \in \mathcal{X}$ to output sequences $y \in \mathcal{Y}$.

Define the datasets:

- $\mathcal{D}^{(t)}_{\text{train}} = \{(q_i, a_i)\}_{i=1}^{N_t}$ — the training dataset at iteration $t$, where $q_i$ is a question and $a_i$ is the expected answer.
- $\mathcal{D}_{\text{val}} = \{(q_j, a_j)\}_{j=1}^{M}$ — the fixed validation dataset.

Partition the training set by answer type:

$$\mathcal{D}^{(t)}_{\text{train}} = \mathcal{D}^{(t)}_{\text{in}} \cup \mathcal{D}^{(t)}_{\text{out}}$$

where:

- $\mathcal{D}^{(t)}_{\text{in}} = \{(q_i, a_i) \mid a_i \neq \phi\}$ — in-domain samples with real answers
- $\mathcal{D}^{(t)}_{\text{out}} = \{(q_i, a_i) \mid a_i = \phi\}$ — out-of-scope samples where $\phi$ is the standard refusal phrase

### 2. LoRA Parameterization

Rather than fine-tuning all $d$ parameters of $\theta$, we learn a low-rank additive perturbation. For each target weight matrix $W_0 \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ in the base model, LoRA decomposes the update as:

$$W = W_0 + \Delta W = W_0 + \frac{\alpha}{r} B A$$

where:

- $A \in \mathbb{R}^{r \times d_{\text{in}}}$ — the down-projection matrix (initialized from $\mathcal{N}(0, \sigma^2)$)
- $B \in \mathbb{R}^{d_{\text{out}} \times r}$ — the up-projection matrix (initialized to zero)
- $r \ll \min(d_{\text{in}}, d_{\text{out}})$ — the rank (default $r = 8$)
- $\alpha$ — the scaling factor (default $\alpha = 16$)

The set of all trainable LoRA parameters is:

$$\Delta\theta = \{(A_l, B_l)\}_{l \in \mathcal{L}}$$

where $\mathcal{L}$ is the set of adapted layers. The total number of trainable parameters is:

$$|\Delta\theta| = |\mathcal{L}| \cdot r \cdot (d_{\text{in}} + d_{\text{out}}) \ll d$$

During training, the forward pass for a given layer becomes:

$$h = (W_0 + \frac{\alpha}{r} B A) x = W_0 x + \frac{\alpha}{r} B(Ax)$$

with dropout applied to $A$ with probability $p_{\text{drop}}$:

$$h = W_0 x + \frac{\alpha}{r} B \cdot \text{Dropout}(Ax, p_{\text{drop}})$$

### 3. Training Objective

The base fine-tuning objective is the causal language modeling (next-token prediction) loss over the formatted training sequences. Let $\mathcal{T}(q, a)$ denote the prompt template function that formats a question-answer pair according to the model's chat template. For a training example $(q_i, a_i)$, the formatted sequence is:

$$s_i = \mathcal{T}(q_i, a_i) = [t_1, t_2, \ldots, t_{L_i}]$$

The standard cross-entropy loss is:

$$\mathcal{L}_{\text{CE}}(\Delta\theta) = - \frac{1}{|\mathcal{D}^{(t)}_{\text{train}}|} \sum_{i=1}^{N_t} \frac{1}{L_i} \sum_{k=1}^{L_i} \log P_{\theta + \Delta\theta}(t_k \mid t_{<k})$$

### 4. Security-Constrained Objective

The framework enforces security through the composition of the training data. The effective optimization objective incorporates three behavioral constraints implicitly through the dataset structure:

**Constraint 1 — Task Fidelity (Topic Specification):**

$$\forall (q_i, a_i) \in \mathcal{D}^{(t)}_{\text{in}}: \quad P_{\theta + \Delta\theta}(a_i \mid q_i) \geq \tau_{\text{task}}$$

The model must produce correct in-domain answers with high probability.

**Constraint 2 — Security (Jailbreak Resistance):**

$$\forall (q_i, \phi) \in \mathcal{D}^{(t)}_{\text{out}}: \quad P_{\theta + \Delta\theta}(\phi \mid q_i) \geq \tau_{\text{sec}}$$

The model must produce the refusal phrase for all out-of-scope queries.

**Constraint 3 — Format Compliance:**

$$\forall (q_i, a_i) \in \mathcal{D}^{(t)}_{\text{train}}: \quad \text{len}(\hat{a}_i) \leq \lambda \cdot \text{len}(a_i)$$

where $\hat{a}_i = \mathcal{M}_{\theta + \Delta\theta}(q_i)$ and $\lambda$ is a format tolerance factor. Answers should be concise.

The combined security-aware objective can be written as:

$$\min_{\Delta\theta} \; \mathcal{L}_{\text{CE}}(\Delta\theta) \quad \text{subject to} \quad \mathcal{C}_{\text{task}} \wedge \mathcal{C}_{\text{sec}} \wedge \mathcal{C}_{\text{fmt}}$$

In practice, these constraints are enforced softly through dataset composition rather than explicit Lagrangian multipliers. The ratio of refusal samples to real answers determines the implicit weight on security:

$$w_{\text{sec}} = \frac{|\mathcal{D}^{(t)}_{\text{out}}|}{|\mathcal{D}^{(t)}_{\text{train}}|}$$

### 5. Optimization Algorithm

The LoRA parameters are optimized using Paged AdamW with cosine learning rate schedule:

$$\Delta\theta^{(s+1)} = \Delta\theta^{(s)} - \eta_s \cdot \frac{\hat{m}_s}{\sqrt{\hat{v}_s} + \epsilon} - \eta_s \lambda_w \Delta\theta^{(s)}$$

where:

- $\eta_s$ — learning rate at step $s$, following cosine decay:

$$\eta_s = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{s - s_w}{S - s_w} \pi\right)\right)$$

- $s_w = \lfloor \rho_w \cdot S \rfloor$ — warmup steps ($\rho_w = 0.03$)
- $\hat{m}_s, \hat{v}_s$ — bias-corrected first and second moment estimates (Adam)
- $\lambda_w$ — weight decay coefficient
- $\epsilon = 10^{-8}$ — numerical stability constant

The effective batch size for gradient computation is:

$$B_{\text{eff}} = B_{\text{device}} \times G_{\text{accum}}$$

where $B_{\text{device}} = 8$ and $G_{\text{accum}} = 4$, yielding $B_{\text{eff}} = 32$.

### 6. NF4 Quantization

The base model weights are quantized to 4-bit NormalFloat (NF4). For a weight tensor $W$, the quantized representation is:

$$\hat{W} = Q_{\text{NF4}}(W) = s \cdot q_k, \quad k = \underset{i}{\arg\min} \; |w / s - c_i|$$

where:

- $s$ — block-wise scaling factor computed per 64-element block
- $\{c_i\}_{i=0}^{15}$ — the 16 NF4 quantization centroids derived from the quantiles of $\mathcal{N}(0, 1)$
- Double quantization: the scaling factors $s$ are themselves quantized to 8-bit

The LoRA computation remains in FP16:

$$h = \hat{W}_0 x + \frac{\alpha}{r} B(Ax) \quad \text{where} \; A, B \in \text{FP16}$$

### 7. GPT-Judge Scoring Function

Define the judge scoring function $\mathcal{J}$, implemented by an external large language model:

$$\mathcal{J}: \mathcal{X} \times \mathcal{Y} \times \mathcal{Y} \rightarrow \{0, 1\}$$

For a validation example $(q_j, a_j)$ with model output $\hat{a}_j$:

$$\mathcal{J}(q_j, a_j, \hat{a}_j) = \begin{cases} 1 & \text{if } \hat{a}_j \text{ is semantically correct w.r.t. } a_j \\ 1 & \text{if } a_j = \phi \text{ and } \hat{a}_j \approx \phi \\ 0 & \text{otherwise} \end{cases}$$

The aggregate validation score at iteration $t$ is:

$$S^{(t)} = \frac{1}{M} \sum_{j=1}^{M} \mathcal{J}(q_j, a_j, \hat{a}_j^{(t)})$$

### 8. ML Evaluation Metrics

From the scored validation records, construct binary classification labels:

$$y_j^{\text{true}} = \begin{cases} 1 & \text{if } a_j \neq \phi \quad \text{(real answer expected)} \\ 0 & \text{if } a_j = \phi \quad \text{(refusal expected)} \end{cases}$$

$$y_j^{\text{pred}} = \begin{cases} y_j^{\text{true}} & \text{if } \mathcal{J}(q_j, a_j, \hat{a}_j) = 1 \\ 1 - y_j^{\text{true}} & \text{if } \mathcal{J}(q_j, a_j, \hat{a}_j) = 0 \end{cases}$$

The confusion matrix entries are:

$$\text{TP} = \sum_j \mathbb{1}[y_j^{\text{true}} = 1 \wedge y_j^{\text{pred}} = 1], \quad \text{TN} = \sum_j \mathbb{1}[y_j^{\text{true}} = 0 \wedge y_j^{\text{pred}} = 0]$$

$$\text{FP} = \sum_j \mathbb{1}[y_j^{\text{true}} = 0 \wedge y_j^{\text{pred}} = 1], \quad \text{FN} = \sum_j \mathbb{1}[y_j^{\text{true}} = 1 \wedge y_j^{\text{pred}} = 0]$$

The computed metrics are:

$$\text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}$$

$$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}, \quad \text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$

$$F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

$$\text{MCC} = \frac{\text{TP} \cdot \text{TN} - \text{FP} \cdot \text{FN}}{\sqrt{(\text{TP}+\text{FP})(\text{TP}+\text{FN})(\text{TN}+\text{FP})(\text{TN}+\text{FN})}}$$

$$\text{ROC-AUC} = \frac{1}{2}\left(\frac{\text{TP}}{\text{TP}+\text{FN}} + \frac{\text{TN}}{\text{TN}+\text{FP}}\right)$$

### 9. Three-Axis Error Analysis

The judge performs structured error analysis over the error set $\mathcal{E}^{(t)} = \{(q_j, a_j, \hat{a}_j) \mid \mathcal{J}(q_j, a_j, \hat{a}_j) = 0\}$.

Define three error indicator functions:

$$e_{\text{fmt}}(q, a, \hat{a}) = \mathbb{1}[\hat{a} \text{ has formatting violations}]$$

$$e_{\text{topic}}(q, a, \hat{a}) = \mathbb{1}[\hat{a} \text{ is off-topic for domain } \mathcal{D}_{\text{domain}}]$$

$$e_{\text{sec}}(q, a, \hat{a}) = \mathbb{1}[a = \phi \wedge \hat{a} \neq \phi] \quad \text{(jailbreak success)}$$

The per-axis error rates are:

$$R_{\text{fmt}}^{(t)} = \frac{1}{|\mathcal{E}^{(t)}|} \sum_{(q,a,\hat{a}) \in \mathcal{E}^{(t)}} e_{\text{fmt}}(q, a, \hat{a})$$

$$R_{\text{topic}}^{(t)} = \frac{1}{|\mathcal{E}^{(t)}|} \sum_{(q,a,\hat{a}) \in \mathcal{E}^{(t)}} e_{\text{topic}}(q, a, \hat{a})$$

$$R_{\text{sec}}^{(t)} = \frac{1}{|\mathcal{E}^{(t)}|} \sum_{(q,a,\hat{a}) \in \mathcal{E}^{(t)}} e_{\text{sec}}(q, a, \hat{a})$$

### 10. Iterative Dataset Refinement

At each iteration $t$, the judge proposes a dataset transformation operator:

$$\mathcal{D}^{(t+1)}_{\text{train}} = \Gamma(\mathcal{D}^{(t)}_{\text{train}}, \mathcal{E}^{(t)}) = \left(\mathcal{D}^{(t)}_{\text{train}} \setminus \mathcal{R}^{(t)} \right) \oplus \mathcal{M}^{(t)} \cup \mathcal{A}^{(t)}$$

where:

- $\mathcal{R}^{(t)} \subseteq \mathcal{D}^{(t)}_{\text{train}}$ — records to **remove** (duplicates, harmful entries)
- $\mathcal{M}^{(t)}$ — **modifications** applied in-place: for index $i \in \mathcal{I}_{\text{mod}}$, replace $(q_i, a_i)$ with $(q_i', a_i')$
- $\mathcal{A}^{(t)} = \{(q_k^{\text{new}}, a_k^{\text{new}})\}_{k=1}^{n_{\text{add}}}$ — new records to **add**, with $n_{\text{add}} \leq 50$

The operator $\oplus$ denotes the in-place modification step, and $\cup$ is set union with duplicate detection:

$$\mathcal{A}^{(t)}_{\text{dedup}} = \{(q, a) \in \mathcal{A}^{(t)} \mid q \notin \{q_i \mid (q_i, a_i) \in \mathcal{D}^{(t)}_{\text{train}} \setminus \mathcal{R}^{(t)}\}\}$$

The dataset size evolves as:

$$N_{t+1} = N_t - |\mathcal{R}^{(t)}| + |\mathcal{A}^{(t)}_{\text{dedup}}|$$

### 11. Security-Preserving Dataset Balance

To maintain security properties across iterations, the refinement must preserve the refusal ratio within bounds:

$$\beta_{\min} \leq \frac{|\mathcal{D}^{(t+1)}_{\text{out}}|}{|\mathcal{D}^{(t+1)}_{\text{train}}|} \leq \beta_{\max}$$

The judge is instructed to generate additions that include both in-domain reinforcement samples and out-of-scope refusal samples, preserving the security posture of the model.

### 12. Convergence and Termination

The iterative loop terminates when any of the following conditions is met:

**Condition 1 — Target Score Reached:**

$$S^{(t)} \geq S_{\text{target}}$$

**Condition 2 — Maximum Iterations Exhausted:**

$$t \geq T_{\max}, \quad T_{\max} \leq 10$$

**Condition 3 — No Errors Found:**

$$|\mathcal{E}^{(t)}| = 0$$

**Condition 4 — No Modifications Proposed:**

$$|\mathcal{R}^{(t)}| + |\mathcal{M}^{(t)}| + |\mathcal{A}^{(t)}| = 0$$

The final output is the model from the best-performing iteration:

$$t^* = \underset{t \in \{1, \ldots, T\}}{\arg\max} \; S^{(t)}$$

$$\mathcal{M}^* = \mathcal{M}_{\theta + \Delta\theta^{(t^*)}}$$

### 13. End-to-End Optimization Summary

The complete framework solves the following bilevel optimization:

**Outer loop** (dataset refinement, $t = 1, \ldots, T_{\max}$):

$$\mathcal{D}^{(t+1)}_{\text{train}} = \underset{\mathcal{D}'}{\arg\max} \; \mathbb{E}_{(q,a) \sim \mathcal{D}_{\text{val}}} \left[\mathcal{J}\left(q, a, \mathcal{M}_{\theta + \Delta\theta^*(\mathcal{D}')}(q)\right)\right]$$

subject to: $|\mathcal{D}' \triangle \mathcal{D}^{(t)}_{\text{train}}| \leq \delta_t$ (bounded perturbation)

**Inner loop** (LoRA optimization for fixed dataset $\mathcal{D}^{(t)}_{\text{train}}$):

$$\Delta\theta^{(t)*} = \underset{\Delta\theta}{\arg\min} \; \mathcal{L}_{\text{CE}}(\Delta\theta; \mathcal{D}^{(t)}_{\text{train}})$$

The outer loop is approximated by the GPT-judge's error analysis and dataset modification proposals, making it tractable without exhaustive combinatorial search over possible dataset configurations. The judge acts as a learned proxy for the gradient of the validation score with respect to the training data:

$$\Gamma \approx \nabla_{\mathcal{D}} S^{(t)}$$

This human-in-the-loop-free approach automates the traditionally manual process of dataset curation and error analysis while maintaining the security constraints throughout all iterations.

---

## Quantization Strategy

All models are loaded with 4-bit NF4 quantization via `bitsandbytes`:

```
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=float16,
    bnb_4bit_use_double_quant=True
)
```

This reduces memory usage by ~4x, allowing 7B models to fit in 8GB VRAM.

---

## LoRA Adapter Architecture

Fine-tuning uses Parameter-Efficient Fine-Tuning (PEFT) with Low-Rank Adaptation (LoRA):

```
Base Model Weights (frozen)
        │
        ▼
   ┌─────────┐
   │ LoRA A  │  (rank r × hidden_dim)
   │ (r=8)   │
   └────┬────┘
        │  ×  alpha/r scaling
   ┌────┴────┐
   │ LoRA B  │  (hidden_dim × rank r)
   │ (r=8)   │
   └────┬────┘
        │
        ▼
  ΔW = B × A  (added to frozen weights during forward pass)
```

After training, `merge_and_unload()` folds LoRA weights into the base model for deployment.

---

## Judge Scoring Pipeline

```
For each validation record:

  ┌──────────────────────────────────────────────────┐
  │ System Prompt: "You are an expert evaluator..."  │
  │                                                  │
  │ User Prompt:                                     │
  │   Question: {question}                           │
  │   Expected answer: {expected}                    │
  │   Model output: {model_output}                   │
  │                                                  │
  │ → LLM responds with: {"score": 0} or {"score": 1}│
  └──────────────────────────────────────────────────┘
```

---

## Error Analysis Pipeline

```
Error records (score=0) ──┐
                          ├──→ Judge LLM analyzes patterns
Correct samples (score=1)─┘         │
Training data sample ──────────────┘
                                    │
                                    ▼
                        ┌──────────────────────┐
                        │  3-Axis Evaluation   │
                        │                      │
                        │  1. Formatting       │
                        │  2. Topic/Domain     │
                        │  3. Security         │
                        │                      │
                        │  → Propose changes:  │
                        │    + additions        │
                        │    ~ modifications    │
                        │    - removals         │
                        └──────────────────────┘
```

---

## File Naming Conventions

| File Pattern | Purpose |
|---|---|
| `training_dataset_iter{N}.jsonl` | Training dataset snapshot at iteration N |
| `training_dataset_final.jsonl` | Final training dataset after all iterations |
| `validation_scored_iter{N}.jsonl` | Validation results with FT responses and scores |
| `metrics_iter{N}.json` | ML metrics for iteration N |
| `error_analysis_iter{N}.json` | Judge error analysis for iteration N |
| `run_summary.json` | Overall run summary with all metrics |
| `iter_{N}/merged_model/` | Merged model files for iteration N |
| `checkpoints_iter{N}/` | Training checkpoints (LoRA adapters) |

---

## Empirical Evaluation Results

This section presents the empirical evaluation of five SLM model pairs — each consisting of a **Fine-Tuned (FT)** variant trained with the Secure Fine-Tune Framework and its corresponding **Out-of-Box (OOB)** baseline — evaluated on a validation dataset of $M = 499$ records ($427$ real-answer positives, $72$ refusal negatives).

All metrics are computed using the binary classification formulation defined in [Section 8](#8-ml-evaluation-metrics), where:

- **Positive class** ($y = 1$): The model should provide a real answer
- **Negative class** ($y = 0$): The model should refuse (produce $\phi$)

The improvement delta for each metric is computed as:

$$\Delta_{\text{metric}} = \text{metric}_{\text{FT}} - \text{metric}_{\text{OOB}}$$

A positive $\Delta$ indicates the fine-tuned model outperforms its baseline.

---

### Model Pair 1: Gemma 2B IT

| | **FT:** `gemma-2b-it-edcastr_JavaScript-v10` | **OOB:** `gemma-2b-it` | **$\Delta$ (FT − OOB)** |
|:---|:---:|:---:|:---:|
| **Accuracy** | 0.6232 | 0.2485 | **+0.3747** |
| **Precision** | 0.9838 | 0.6327 | **+0.3511** |
| **Recall** | 0.5691 | 0.2904 | **+0.2787** |
| **$F_1$ Score** | 0.7211 | 0.3981 | **+0.3230** |
| **MCC** | 0.3609 | −0.5106 | **+0.8715** |
| **ROC-AUC** | 0.7568 | 0.1452 | **+0.6116** |

**Confusion Matrices:**

| | **FT — Gemma 2B** | | | **OOB — Gemma 2B** | |
|:---|:---:|:---:|:---|:---:|:---:|
| | Pred Refused (0) | Pred Answer (1) | | Pred Refused (0) | Pred Answer (1) |
| **Actual Refused (0)** | TN = 68 | FP = 4 | **Actual Refused (0)** | TN = 0 | FP = 72 |
| **Actual Answer (1)** | FN = 184 | TP = 243 | **Actual Answer (1)** | FN = 303 | TP = 124 |

> **Key Finding:** The OOB Gemma model performs poorly across both classes: it fails to refuse any out-of-scope queries ($\text{TN} = 0$, $\text{FP} = 72$) and answers only 124 of 427 real-answer questions correctly ($\text{MCC} = -0.5106$). After fine-tuning, refusal detection improves substantially ($\text{TN} = 68$, $\text{FP} = 4$, 94.4% refusal accuracy), though the model over-refuses on real-answer queries ($\text{FN} = 184$), resulting in moderate overall performance ($F_1 = 0.7211$, $\text{MCC} = 0.3609$).

---

### Model Pair 2: TinyLlama 1.1B Chat

| | **FT:** `tinyllama-edcastr_JavaScript-v2` | **OOB:** `TinyLlama-1.1B-Chat-v1.0` | **$\Delta$ (FT − OOB)** |
|:---|:---:|:---:|:---:|
| **Accuracy** | 0.7034 | 0.2084 | **+0.4950** |
| **Precision** | 1.0000 | 0.5909 | **+0.4091** |
| **Recall** | 0.6534 | 0.2436 | **+0.4098** |
| **$F_1$ Score** | 0.7904 | 0.3449 | **+0.4455** |
| **MCC** | 0.4624 | −0.5563 | **+1.0187** |
| **ROC-AUC** | 0.8267 | 0.1218 | **+0.7049** |

**Confusion Matrices:**

| | **FT — TinyLlama** | | | **OOB — TinyLlama** | |
|:---|:---:|:---:|:---|:---:|:---:|
| | Pred Refused (0) | Pred Answer (1) | | Pred Refused (0) | Pred Answer (1) |
| **Actual Refused (0)** | TN = 72 | FP = 0 | **Actual Refused (0)** | TN = 0 | FP = 72 |
| **Actual Answer (1)** | FN = 148 | TP = 279 | **Actual Answer (1)** | FN = 323 | TP = 104 |

> **Key Finding:** The OOB TinyLlama model performs worse than random ($\text{ROC-AUC} = 0.1218$, $\text{MCC} = -0.5563$), indicating systematic anti-correlation with the correct labels. Fine-tuning achieves perfect refusal detection ($\text{FP} = 0$, $\text{TN} = 72$) and dramatically improves answer quality, producing the largest MCC swing among all models ($\Delta_{\text{MCC}} = +1.0187$).

---

### Model Pair 3: DeepSeek R1 Distill Qwen 1.5B

| | **FT:** `DeepSeek-R1-Distill-Qwen-1.5B-LoRA` | **OOB:** `DeepSeek-R1-Distill-Qwen-1.5B` | **$\Delta$ (FT − OOB)** |
|:---|:---:|:---:|:---:|
| **Accuracy** | 0.7515 | 0.0822 | **+0.6693** |
| **Precision** | 1.0000 | 0.3628 | **+0.6372** |
| **Recall** | 0.7096 | 0.0960 | **+0.6136** |
| **$F_1$ Score** | 0.8301 | 0.1519 | **+0.6782** |
| **MCC** | 0.5106 | −0.7589 | **+1.2695** |
| **ROC-AUC** | 0.8548 | 0.0480 | **+0.8068** |

**Confusion Matrices:**

| | **FT — DeepSeek** | | | **OOB — DeepSeek** | |
|:---|:---:|:---:|:---|:---:|:---:|
| | Pred Refused (0) | Pred Answer (1) | | Pred Refused (0) | Pred Answer (1) |
| **Actual Refused (0)** | TN = 72 | FP = 0 | **Actual Refused (0)** | TN = 0 | FP = 72 |
| **Actual Answer (1)** | FN = 124 | TP = 303 | **Actual Answer (1)** | FN = 386 | TP = 41 |

> **Key Finding:** The OOB DeepSeek model exhibits the strongest anti-correlation of all baselines ($\text{MCC} = -0.7589$, $\text{ROC-AUC} = 0.048$), answering correctly only 41 out of 427 real-answer questions while failing every refusal. Fine-tuning transforms the model from near-total failure to solid performance with perfect refusal detection ($\text{FP} = 0$) and the largest $\Delta_{\text{MCC}} = +1.2695$ observed across all pairs.

---

### Model Pair 4: Phi-3 Mini 4K Instruct

| | **FT:** `Phi_3_mini_4k_instruct_LoRA` | **OOB:** `Phi-3-mini-4k-Instruct` | **$\Delta$ (FT − OOB)** |
|:---|:---:|:---:|:---:|
| **Accuracy** | 0.8096 | 0.3287 | **+0.4809** |
| **Precision** | 1.0000 | 0.6949 | **+0.3051** |
| **Recall** | 0.7775 | 0.3841 | **+0.3934** |
| **$F_1$ Score** | 0.8748 | 0.4947 | **+0.3801** |
| **MCC** | 0.5790 | −0.4335 | **+1.0125** |
| **ROC-AUC** | 0.8888 | 0.1920 | **+0.6968** |

**Confusion Matrices:**

| | **FT — Phi-3** | | | **OOB — Phi-3** | |
|:---|:---:|:---:|:---|:---:|:---:|
| | Pred Refused (0) | Pred Answer (1) | | Pred Refused (0) | Pred Answer (1) |
| **Actual Refused (0)** | TN = 72 | FP = 0 | **Actual Refused (0)** | TN = 0 | FP = 72 |
| **Actual Answer (1)** | FN = 95 | TP = 332 | **Actual Answer (1)** | FN = 263 | TP = 164 |

> **Key Finding:** Phi-3 OOB has the best baseline among the scored models ($\text{Accuracy} = 0.3287$, $\text{TP} = 164$) but still completely fails at refusal. Fine-tuning yields the highest FT accuracy among scored models ($0.8096$) and the highest FT recall ($0.7775$), achieving perfect refusal performance ($\text{FP} = 0$) with a $\Delta_{\text{MCC}} = +1.0125$.

---

### Model Pair 5: Qwen 1.5 0.5B Chat

| | **FT:** `Qwen1.5-0.5B-Chat-edcastr_JavaScript-v1` | **OOB:** `Qwen1.5-0.5B-Chat` | **$\Delta$ (FT − OOB)** |
|:---|:---:|:---:|:---:|
| **Accuracy** | 0.7976 | 0.1182 | **+0.6794** |
| **Precision** | 1.0000 | 0.4504 | **+0.5496** |
| **Recall** | 0.7635 | 0.1382 | **+0.6253** |
| **$F_1$ Score** | 0.8659 | 0.2115 | **+0.6544** |
| **MCC** | 0.5637 | −0.6882 | **+1.2519** |
| **ROC-AUC** | 0.8817 | 0.0691 | **+0.8126** |

**Confusion Matrices:**

| | **FT — Qwen 1.5** | | | **OOB — Qwen 1.5** | |
|:---|:---:|:---:|:---|:---:|:---:|
| | Pred Refused (0) | Pred Answer (1) | | Pred Refused (0) | Pred Answer (1) |
| **Actual Refused (0)** | TN = 72 | FP = 0 | **Actual Refused (0)** | TN = 0 | FP = 72 |
| **Actual Answer (1)** | FN = 101 | TP = 326 | **Actual Answer (1)** | FN = 368 | TP = 59 |

> **Key Finding:** Qwen OOB is the second-worst baseline ($\text{Accuracy} = 0.1182$, $\text{MCC} = -0.6882$), answering only 59 of 427 real-answer questions correctly. Fine-tuning produces the second-largest overall improvement ($\Delta_{\text{MCC}} = +1.2519$), achieving perfect refusal detection and an accuracy of $0.7976$.

---

### Cross-Model Comparative Summary

The table below consolidates all metrics across the five model pairs, enabling direct comparison:

| Model Pair | Mode | Accuracy | Precision | Recall | $F_1$ | MCC | ROC-AUC |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Gemma 2B** | FT | 0.6232 | 0.9838 | 0.5691 | 0.7211 | 0.3609 | 0.7568 |
| | OOB | 0.2485 | 0.6327 | 0.2904 | 0.3981 | −0.5106 | 0.1452 |
| | $\Delta$ | **+0.3747** | **+0.3511** | **+0.2787** | **+0.3230** | **+0.8715** | **+0.6116** |
| **TinyLlama 1.1B** | FT | 0.7034 | 1.0000 | 0.6534 | 0.7904 | 0.4624 | 0.8267 |
| | OOB | 0.2084 | 0.5909 | 0.2436 | 0.3449 | −0.5563 | 0.1218 |
| | $\Delta$ | **+0.4950** | **+0.4091** | **+0.4098** | **+0.4455** | **+1.0187** | **+0.7049** |
| **DeepSeek 1.5B** | FT | 0.7515 | 1.0000 | 0.7096 | 0.8301 | 0.5106 | 0.8548 |
| | OOB | 0.0822 | 0.3628 | 0.0960 | 0.1519 | −0.7589 | 0.0480 |
| | $\Delta$ | **+0.6693** | **+0.6372** | **+0.6136** | **+0.6782** | **+1.2695** | **+0.8068** |
| **Phi-3 Mini** | FT | 0.8096 | 1.0000 | 0.7775 | 0.8748 | 0.5790 | 0.8888 |
| | OOB | 0.3287 | 0.6949 | 0.3841 | 0.4947 | −0.4335 | 0.1920 |
| | $\Delta$ | **+0.4809** | **+0.3051** | **+0.3934** | **+0.3801** | **+1.0125** | **+0.6968** |
| **Qwen 1.5 0.5B** | FT | 0.7976 | 1.0000 | 0.7635 | 0.8659 | 0.5637 | 0.8817 |
| | OOB | 0.1182 | 0.4504 | 0.1382 | 0.2115 | −0.6882 | 0.0691 |
| | $\Delta$ | **+0.6794** | **+0.5496** | **+0.6253** | **+0.6544** | **+1.2519** | **+0.8126** |

---

### Executive Summary

The empirical evaluation across five SLM model pairs validates the effectiveness of the Secure Fine-Tune Framework on both axes of the security-constrained objective: **task fidelity** (correct answers to in-domain questions) and **jailbreak resistance** (refusal of out-of-scope queries).

#### 1. Universal Security Improvement

Every OOB baseline model achieves $\text{TN} = 0$ — none of the five pre-trained models can refuse out-of-scope queries without fine-tuning. All queries, including jailbreak attempts, receive an answer, resulting in $\text{FP} = 72$ (100% false-positive rate on the refusal class). After fine-tuning:

- **Four out of five FT models achieve perfect refusal** ($\text{FP} = 0$, $\text{TN} = 72$): TinyLlama, DeepSeek, Phi-3, and Qwen.
- **Gemma FT achieves near-perfect refusal** ($\text{FP} = 4$, $\text{TN} = 68$), with a 94.4% refusal accuracy.

This demonstrates that the framework's security constraint ($\mathcal{C}_{\text{sec}}$) is effectively enforced through dataset composition.

#### 2. Task Fidelity Gains

Fine-tuning simultaneously improves real-answer quality:

| Model | OOB $F_1$ | FT $F_1$ | $\Delta F_1$ |
|:---|:---:|:---:|:---:|
| Gemma 2B | 0.3981 | 0.7211 | +0.3230 |
| TinyLlama 1.1B | 0.3449 | 0.7904 | +0.4455 |
| DeepSeek 1.5B | 0.1519 | 0.8301 | +0.6782 |
| Phi-3 Mini | 0.4947 | 0.8748 | +0.3801 |
| Qwen 1.5 0.5B | 0.2115 | 0.8659 | +0.6544 |

The mean $F_1$ improvement across all models is:

$$\overline{\Delta F_1} = \frac{1}{5}\sum_{i=1}^{5} \Delta F_{1,i} = \frac{0.3230 + 0.4455 + 0.6782 + 0.3801 + 0.6544}{5} = 0.4962$$

#### 3. Matthews Correlation Coefficient Analysis

MCC is the most informative single metric for imbalanced binary classification. The OOB baselines all exhibit $\text{MCC} \leq 0$, confirming they have no positive discriminative ability for the refusal task. The mean MCC improvement is:

$$\overline{\Delta\text{MCC}} = \frac{0.8715 + 1.0187 + 1.2695 + 1.0125 + 1.2519}{5} = 1.0848$$

This means fine-tuning shifts models from anti-correlated or random classification to strong positive correlation with the correct labels.

#### 4. Model Ranking

Ranked by post-fine-tuning $F_1$ score:

| Rank | Model (FT) | $F_1$ | MCC | Accuracy |
|:---:|:---|:---:|:---:|:---:|
| 1 | Phi-3 Mini 4K | 0.8748 | 0.5790 | 0.8096 |
| 2 | Qwen 1.5 0.5B | 0.8659 | 0.5637 | 0.7976 |
| 3 | DeepSeek R1 1.5B | 0.8301 | 0.5106 | 0.7515 |
| 4 | TinyLlama 1.1B | 0.7904 | 0.4624 | 0.7034 |
| 5 | Gemma 2B IT | 0.7211 | 0.3609 | 0.6232 |

Phi-3 Mini 4K leads the fine-tuned models with the highest $F_1 = 0.8748$. The top four models cluster between $F_1 \in [0.79, 0.87]$, while Gemma 2B IT ranks last ($F_1 = 0.7211$) due to a high false-negative rate ($\text{FN} = 184$), indicating the model over-refuses real-answer queries. Despite this, all five models show substantial improvement over their OOB baselines, confirming that the framework consistently produces security-constrained models regardless of the base architecture, with room for further improvement through additional iterations of the judge-in-the-loop refinement process.

#### 5. Security ROC-AUC Interpretation

From a security perspective, ROC-AUC measures the model's ability to distinguish between queries it should answer and queries it should refuse:

$$\text{ROC-AUC} = \frac{1}{2}\left(\frac{\text{TP}}{\text{TP}+\text{FN}} + \frac{\text{TN}}{\text{TN}+\text{FP}}\right)$$

All OOB models achieve $\text{ROC-AUC} \leq 0.5$ (at or worse than random), while all FT models achieve $\text{ROC-AUC} \geq 0.7568$, confirming reliable jailbreak detection. The mean improvement is:

$$\overline{\Delta\text{ROC-AUC}} = \frac{0.6116 + 0.7049 + 0.8068 + 0.6968 + 0.8126}{5} = 0.7265$$
