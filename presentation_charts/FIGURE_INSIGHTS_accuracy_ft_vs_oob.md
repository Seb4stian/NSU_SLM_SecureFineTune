# Key Takeaways: Task-Specific Accuracy — OOB vs Fine-Tuned (11 Models)

## Figure: slide12_accuracy_ft_vs_oob.png

The figure presents a comparative bar chart of task-specific accuracy (JavaScript library Q&A) across eleven Small Language Models, contrasting their Out-of-Box (OOB) baseline performance against their fine-tuned (FT) counterparts produced by the Secure Fine-Tune Framework.

## Key Insights

The results demonstrate a **universal and substantial improvement** in task accuracy across all eleven models after fine-tuning, with accuracy gains (Δ) ranging from +0.37 to +0.79. Every OOB baseline performs poorly on the domain-specific task — no model exceeds 33% accuracy without fine-tuning — confirming that pretrained SLMs lack the specialized knowledge required for focused Q&A tasks without targeted adaptation. After fine-tuning, all models surpass 62% accuracy, with the top three (OLMo 7B at 94%, StableLM-2 at 91%, and H2O-Danube at 90%) approaching near-expert performance.

A clear **scale-performance relationship** emerges: larger models (OLMo 7B, StableLM-2 1.6B, H2O-Danube 1.8B) achieve the highest post-fine-tuning accuracy (90–94%), while the smallest model (SmolLM2 at 135M parameters) reaches only 65%, suggesting diminishing returns below ~1B parameters. However, even mid-range models like MiniCPM (1B), Phi-3 Mini (3.8B), and Qwen 1.5 (0.5B) achieve strong performance in the 80% range, demonstrating that the framework is effective across diverse architectures and scales.

Notably, the **largest relative improvements** (Δ ≥ +0.67) occur in models with the weakest OOB baselines — DeepSeek (8% → 75%, Δ = +0.67), Qwen (12% → 80%, Δ = +0.68), H2O-Danube (11% → 90%, Δ = +0.79), and OLMo (18% → 94%, Δ = +0.76) — indicating that the framework is most impactful precisely where the pretrained model has the least pre-existing domain knowledge. This validates the core premise of the dissertation: iterative fine-tuning with LLM-judge feedback can reliably transform underperforming SLMs into competent domain-specific assistants, regardless of their initial capability level.
