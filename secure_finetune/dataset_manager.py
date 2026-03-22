"""
Dataset manager: load, modify, and save JSONL datasets.

Handles dataset operations including additions, modifications,
and removals based on GPT-judge feedback.
"""

import json
import os
import copy
import random
from typing import Optional


class DatasetManager:
    """Manages training and validation JSONL datasets."""

    def __init__(self, refusal_phrase: str = "I am sorry, I do not have an answer for your question"):
        self.refusal_phrase = refusal_phrase

    def load_jsonl(self, filepath: str) -> list[dict]:
        """Load a JSONL file into a list of records."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dataset file not found: {filepath}")

        records = []
        with open(filepath, "r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    print(f"  ⚠ Skipping line {line_no}: {exc}")
        return records

    def save_jsonl(self, records: list[dict], filepath: str):
        """Save records to a JSONL file."""
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def get_dataset_stats(self, records: list[dict]) -> dict:
        """Compute statistics about the dataset."""
        total = len(records)
        refusals = sum(
            1 for r in records
            if self.refusal_phrase.lower() in r.get("Answer", "").lower()
        )
        return {
            "total": total,
            "real_answers": total - refusals,
            "refusals": refusals,
            "refusal_ratio": refusals / total if total > 0 else 0,
        }

    def apply_modifications(
        self,
        records: list[dict],
        additions: list[dict],
        modifications: list[dict],
        removals: list[int],
    ) -> list[dict]:
        """
        Apply judge-proposed modifications to the dataset.

        Args:
            records: Current dataset records
            additions: New Q&A pairs to add [{"Question": ..., "Answer": ...}, ...]
            modifications: Records to modify [{"index": int, "Question": ..., "Answer": ...}, ...]
            removals: Indices of records to remove [int, ...]

        Returns:
            Modified dataset records
        """
        result = copy.deepcopy(records)

        # Apply removals (in reverse order to preserve indices)
        for idx in sorted(removals, reverse=True):
            if 0 <= idx < len(result):
                result.pop(idx)

        # Apply modifications
        for mod in modifications:
            idx = mod.get("index", -1)
            if 0 <= idx < len(result):
                if "Question" in mod:
                    result[idx]["Question"] = mod["Question"]
                if "Answer" in mod:
                    result[idx]["Answer"] = mod["Answer"]

        # Apply additions
        for addition in additions:
            if "Question" in addition and "Answer" in addition:
                result.append({
                    "Question": addition["Question"],
                    "Answer": addition["Answer"],
                })

        return result

    def create_versioned_path(self, original_path: str, iteration: int) -> str:
        """Create a versioned file path for the dataset."""
        base, ext = os.path.splitext(original_path)
        return f"{base}_iter{iteration}{ext}"

    def sample_records(
        self, records: list[dict], n: int = 50, balanced: bool = True
    ) -> list[dict]:
        """
        Sample records from the dataset for judge analysis.

        If balanced=True, tries to get equal numbers of refusals and real answers.
        """
        if len(records) <= n:
            return records

        if not balanced:
            return random.sample(records, n)

        refusals = [
            r for r in records
            if self.refusal_phrase.lower() in r.get("Answer", "").lower()
        ]
        answers = [
            r for r in records
            if self.refusal_phrase.lower() not in r.get("Answer", "").lower()
        ]

        half = n // 2
        sampled_refusals = random.sample(refusals, min(half, len(refusals)))
        sampled_answers = random.sample(answers, min(n - len(sampled_refusals), len(answers)))

        result = sampled_refusals + sampled_answers
        random.shuffle(result)
        return result

    def merge_error_samples(
        self,
        error_records: list[dict],
        existing_records: list[dict],
    ) -> list[dict]:
        """
        Merge reinforcement samples into the training dataset,
        avoiding exact duplicates.
        """
        existing_questions = {r.get("Question", "").strip().lower() for r in existing_records}
        new_records = []

        for rec in error_records:
            q = rec.get("Question", "").strip().lower()
            if q and q not in existing_questions:
                new_records.append(rec)
                existing_questions.add(q)

        return existing_records + new_records
