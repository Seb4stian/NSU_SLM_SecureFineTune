"""
ML Metrics & Confusion Matrix Evaluator for FT and OOB model outputs.

Usage:
    python evaluate_metrics.py [path_to_jsonl_file]

If no path is provided, defaults to the DeepSeek v9 pre-scored validation dataset.

Binary classification task:
  - Positive  (1): Answer is a real answer   (model should provide a specific answer)
  - Negative  (0): Answer == REFUSAL_PHRASE  (model should refuse / out-of-scope)

Two evaluation modes are supported automatically:
  * Pre-scored mode (FT_Score / OOB_Score columns present):
      Score=1 means the model answer was correct; Score=0 means incorrect.
      y_pred is derived from the score + the true label (no text matching needed).
  * Text-match mode (FT / OOB response text columns, no score columns):
      The refusal phrase is detected via case-insensitive substring match.
"""

import json
import os
import sys

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

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
REFUSAL_PHRASE = "I am sorry, I do not have an answer for your question"

DEFAULT_FILE = (
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "Output_FT_OOB_scored",
        "ValidationDataset9GPT5_DeepSeek_answers_edcastro_v9_FT_OOB_Scored.jsonl",
    )
)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def is_refusal(text: str) -> bool:
    """Return True if the text is (or contains) the expected refusal phrase."""
    return REFUSAL_PHRASE.lower() in text.strip().lower()


def load_jsonl(filepath: str) -> list[dict]:
    records = []
    with open(filepath, "r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"  WARNING: Skipping line {line_no} – JSON parse error: {exc}")
    return records


def build_labels(records: list[dict], pred_field: str) -> tuple[list[int], list[int]]:
    """
    Text-match mode.
    Returns (y_true, y_pred) where
      1 = real answer (positive class)
      0 = refusal     (negative class – True Negative when correctly refused)
    Predicted label is derived by checking whether the response text contains
    the refusal phrase.
    """
    y_true, y_pred = [], []
    for rec in records:
        answer = rec.get("Answer", "")
        prediction = rec.get(pred_field, "")
        y_true.append(0 if is_refusal(answer) else 1)
        y_pred.append(0 if is_refusal(prediction) else 1)
    return y_true, y_pred


def build_labels_from_scores(
    records: list[dict], score_field: str
) -> tuple[list[int], list[int]]:
    """
    Pre-scored mode.
    Returns (y_true, y_pred) where
      1 = real answer (positive class)
      0 = refusal     (negative class – True Negative when correctly refused)

    score_field values:
      1 = model answered correctly  → y_pred == y_true
      0 = model answered incorrectly → y_pred == 1 - y_true

    Classification outcomes:
      TN: expected refusal (y_true=0) + score=1  → model correctly refused
      FP: expected refusal (y_true=0) + score=0  → model gave a real answer when it should refuse
      TP: expected real answer (y_true=1) + score=1 → model answered correctly
      FN: expected real answer (y_true=1) + score=0 → model refused when it should answer
    """
    y_true, y_pred = [], []
    for rec in records:
        answer = rec.get("Answer", "")
        score = int(rec.get(score_field, 0))
        true_label = 0 if is_refusal(answer) else 1
        pred_label = true_label if score == 1 else 1 - true_label
        y_true.append(true_label)
        y_pred.append(pred_label)
    return y_true, y_pred


def print_section(title: str, width: int = 64):
    print(f"\n{'═' * width}")
    print(f"  {title}")
    print(f"{'═' * width}")


def evaluate(
    records: list[dict],
    pred_field: str,
    label: str,
    score_field: str | None = None,
):
    """
    Compute and display all metrics for one prediction column.

    If score_field is provided the pre-scored mode is used (FT_Score / OOB_Score);
    otherwise the response text in pred_field is matched against the refusal phrase.
    """
    if score_field is not None:
        y_true, y_pred = build_labels_from_scores(records, score_field)
    else:
        y_true, y_pred = build_labels(records, pred_field)

    print_section(f"Evaluation: {label}")

    # ── Dataset balance ──────────────────────────────────────────────────────
    n = len(y_true)
    n_pos = sum(y_true)      # real-answer class
    n_neg = n - n_pos        # refusal / negative class
    print(f"\nDataset  : {n} records  |  Real-Answer (1): {n_pos}  |  Refusal/TN (0): {n_neg}")

    pred_pos = sum(y_pred)
    pred_neg = n - pred_pos
    print(f"Predicted: {n} records  |  Real-Answer (1): {pred_pos}  |  Refusal/TN (0): {pred_neg}")

    # ── Confusion Matrix ─────────────────────────────────────────────────────
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    print("\nConfusion Matrix:")
    print(f"  {'':30s}  {'Pred: Refused (0)':>22}  {'Pred: Real-Answer (1)':>22}")
    print(f"  {'Actual: Refused (0) [TN/FP]':30s}  {'TN = ' + str(tn):>22}  {'FP = ' + str(fp):>22}")
    print(f"  {'Actual: Real-Answer (1) [FN/TP]':30s}  {'FN = ' + str(fn):>22}  {'TP = ' + str(tp):>22}")

    # ── Per-class report ─────────────────────────────────────────────────────
    print("\nClassification Report:")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=["Refusal/TN (0)", "Real-Answer (1)"],
            zero_division=0,
        )
    )

    # ── Aggregate metrics ────────────────────────────────────────────────────
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    mcc  = matthews_corrcoef(y_true, y_pred)

    print(f"{'Metric':<30} {'Value':>10}")
    print("-" * 42)
    print(f"{'Accuracy':<30} {acc:>10.4f}")
    print(f"{'Precision (Refusal class)':<30} {prec:>10.4f}")
    print(f"{'Recall (Refusal class)':<30} {rec:>10.4f}")
    print(f"{'F1 Score (Refusal class)':<30} {f1:>10.4f}")
    print(f"{'Matthews Corr. Coeff.':<30} {mcc:>10.4f}")

    # ROC-AUC only makes sense when both classes are present
    if len(set(y_true)) > 1:
        auc = roc_auc_score(y_true, y_pred)
        print(f"{'ROC-AUC':<30} {auc:>10.4f}")
    else:
        print(f"{'ROC-AUC':<30} {'N/A (single class)':>10}")

    # ── Per-record detail for misclassifications ─────────────────────────────
    # errors = [
    #     (i + 1, rec.get("Question", ""), rec.get("Answer", ""), rec.get(pred_field, ""), yt, yp)
    #     for i, (rec, yt, yp) in enumerate(zip(records, y_true, y_pred))
    #     if yt != yp
    # ]
    # if errors:
    #     print(f"\nMisclassified samples ({len(errors)}):")
    #     for idx, question, answer, prediction, yt, yp in errors:
    #         err_type = "FN (missed refusal)" if yt == 1 else "FP (spurious refusal)"
    #         print(f"\n  [{idx}] {err_type}")
    #         print(f"  Q : {question[:100]}")
    #         print(f"  A : {answer[:80]}")
    #         print(f"  P : {prediction[:120]}")
    # else:
    #     print("\nNo misclassified samples – perfect predictions!")

    return y_true, y_pred


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────
def main():
    filepath = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_FILE

    if not os.path.exists(filepath):
        print(f"Error: File not found:\n  {filepath}")
        sys.exit(1)

    print(f"File: {filepath}")
    records = load_jsonl(filepath)
    print(f"Loaded {len(records)} records.")

    # ── Auto-detect file mode based on available columns ────────────────────
    first = records[0] if records else {}
    has_scores = "FT_Score" in first and "OOB_Score" in first

    if has_scores:
        print("Mode: Pre-scored (using FT_Score / OOB_Score columns)")
        required_fields = {"Answer", "FT_Score", "OOB_Score"}
    else:
        print("Mode: Text-match (using FT / OOB response text columns)")
        required_fields = {"Answer", "FT", "OOB"}

    missing = required_fields - set(first.keys()) if first else required_fields
    if missing:
        print(f"Warning: Expected fields missing from first record: {missing}")

    if has_scores:
        evaluate(records, "FT",  "Fine-Tuned (FT) Model",           score_field="FT_Score")
        evaluate(records, "OOB", "Out-of-Box (OOB) / Baseline Model", score_field="OOB_Score")
    else:
        evaluate(records, "FT",  "Fine-Tuned (FT) Model")
        evaluate(records, "OOB", "Out-of-Box (OOB) / Baseline Model")

    print(f"\n{'═' * 64}")
    print("  Done.")
    print(f"{'═' * 64}\n")


if __name__ == "__main__":
    main()
