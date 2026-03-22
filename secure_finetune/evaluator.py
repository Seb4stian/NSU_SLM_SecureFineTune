"""
Evaluation module: ML metrics computation for fine-tuned model outputs.

Wraps the logic from evaluate_metrics.py for use in the iterative
fine-tuning loop.
"""

import json
import os
from typing import Optional


def _import_sklearn():
    import numpy as np
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        confusion_matrix,
        classification_report,
        roc_auc_score,
        matthews_corrcoef,
    )
    return {
        "np": np,
        "accuracy_score": accuracy_score,
        "precision_score": precision_score,
        "recall_score": recall_score,
        "f1_score": f1_score,
        "confusion_matrix": confusion_matrix,
        "classification_report": classification_report,
        "roc_auc_score": roc_auc_score,
        "matthews_corrcoef": matthews_corrcoef,
    }


class Evaluator:
    """Compute ML metrics for model evaluation."""

    def __init__(
        self,
        refusal_phrase: str = "I am sorry, I do not have an answer for your question",
    ):
        self.refusal_phrase = refusal_phrase

    def is_refusal(self, text: str) -> bool:
        """Check if text contains the refusal phrase."""
        return self.refusal_phrase.lower() in text.strip().lower()

    def build_labels_from_scores(
        self, records: list[dict], score_field: str = "FT_Score"
    ) -> tuple[list[int], list[int]]:
        """
        Build y_true/y_pred from pre-scored records.

        score=1 means model was correct → y_pred == y_true
        score=0 means model was incorrect → y_pred == 1 - y_true
        """
        y_true, y_pred = [], []
        for rec in records:
            answer = rec.get("Answer", "")
            score = int(rec.get(score_field, 0))
            true_label = 0 if self.is_refusal(answer) else 1
            pred_label = true_label if score == 1 else 1 - true_label
            y_true.append(true_label)
            y_pred.append(pred_label)
        return y_true, y_pred

    def build_labels_from_text(
        self, records: list[dict], pred_field: str = "FT"
    ) -> tuple[list[int], list[int]]:
        """
        Build y_true/y_pred from response text (no scores).

        Checks if the response text contains the refusal phrase.
        """
        y_true, y_pred = [], []
        for rec in records:
            answer = rec.get("Answer", "")
            prediction = rec.get(pred_field, "")
            y_true.append(0 if self.is_refusal(answer) else 1)
            y_pred.append(0 if self.is_refusal(prediction) else 1)
        return y_true, y_pred

    def compute_metrics(
        self,
        records: list[dict],
        score_field: Optional[str] = "FT_Score",
        pred_field: str = "FT",
    ) -> dict:
        """
        Compute all ML metrics for the given records.

        Returns a dict with all computed metrics.
        """
        sk = _import_sklearn()

        if score_field and score_field in records[0]:
            y_true, y_pred = self.build_labels_from_scores(records, score_field)
        else:
            y_true, y_pred = self.build_labels_from_text(records, pred_field)

        n = len(y_true)
        n_pos = sum(y_true)
        n_neg = n - n_pos

        cm = sk["confusion_matrix"](y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        acc = sk["accuracy_score"](y_true, y_pred)
        prec = sk["precision_score"](y_true, y_pred, zero_division=0)
        rec = sk["recall_score"](y_true, y_pred, zero_division=0)
        f1 = sk["f1_score"](y_true, y_pred, zero_division=0)
        mcc = sk["matthews_corrcoef"](y_true, y_pred)

        metrics = {
            "total_records": n,
            "real_answers": n_pos,
            "refusals": n_neg,
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "mcc": round(mcc, 4),
            "confusion_matrix": {
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            },
        }

        if len(set(y_true)) > 1:
            metrics["roc_auc"] = round(sk["roc_auc_score"](y_true, y_pred), 4)
        else:
            metrics["roc_auc"] = None

        # Compute average score directly
        if score_field and records and score_field in records[0]:
            scores = [int(r.get(score_field, 0)) for r in records]
            metrics["avg_score"] = round(sum(scores) / len(scores), 4)

        return metrics

    def print_metrics(self, metrics: dict, label: str = "Evaluation"):
        """Print formatted metrics to console."""
        print(f"\n{'═' * 64}")
        print(f"  {label}")
        print(f"{'═' * 64}")

        print(f"\n  Dataset: {metrics['total_records']} records")
        print(f"    Real-Answer (1): {metrics['real_answers']}")
        print(f"    Refusal     (0): {metrics['refusals']}")

        cm = metrics["confusion_matrix"]
        print(f"\n  Confusion Matrix:")
        print(f"    {'':28s}  {'Pred Refused':>14}  {'Pred Answer':>14}")
        print(f"    {'Actual Refused [TN/FP]':28s}  {'TN=' + str(cm['tn']):>14}  {'FP=' + str(cm['fp']):>14}")
        print(f"    {'Actual Answer  [FN/TP]':28s}  {'FN=' + str(cm['fn']):>14}  {'TP=' + str(cm['tp']):>14}")

        print(f"\n  {'Metric':<28} {'Value':>10}")
        print(f"  {'-' * 40}")
        print(f"  {'Accuracy':<28} {metrics['accuracy']:>10.4f}")
        print(f"  {'Precision':<28} {metrics['precision']:>10.4f}")
        print(f"  {'Recall':<28} {metrics['recall']:>10.4f}")
        print(f"  {'F1 Score':<28} {metrics['f1_score']:>10.4f}")
        print(f"  {'MCC':<28} {metrics['mcc']:>10.4f}")
        if metrics.get("roc_auc") is not None:
            print(f"  {'ROC-AUC':<28} {metrics['roc_auc']:>10.4f}")
        if metrics.get("avg_score") is not None:
            print(f"  {'Avg Judge Score':<28} {metrics['avg_score']:>10.4f}")
        print()

    def get_error_records(
        self, records: list[dict], score_field: str = "FT_Score"
    ) -> tuple[list[dict], list[dict]]:
        """Split records into errors (score=0) and correct (score=1)."""
        errors = [r for r in records if int(r.get(score_field, 0)) == 0]
        correct = [r for r in records if int(r.get(score_field, 0)) == 1]
        return errors, correct

    def save_metrics(self, metrics: dict, filepath: str):
        """Save metrics to a JSON file."""
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
