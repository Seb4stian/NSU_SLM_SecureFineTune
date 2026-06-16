"""
Score ablation analysis results (OOB, FT, OOB_ToxicSuppressed) using OpenAI GPT-5.1.
Produces scored JSONL files and a scorecard with statistics.
"""

import os
import json
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path
import scipy.stats as stats
import time
import sys
import threading

# Load API key from the Output_FT_OOB_scored/.env file
ENV_PATH = Path(__file__).parent.parent / "Output_FT_OOB_scored" / ".env"
load_dotenv(ENV_PATH)
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY or API_KEY == "your-api-key-here":
    print("ERROR: Please set your OpenAI API key in the .env file")
    sys.exit(1)

OUTPUT_DIR = Path(__file__).parent
PROMPT_FILE = Path(__file__).parent.parent / "Output_FT_OOB_scored" / "prompts" / "scoring_system_prompt.txt"
if not PROMPT_FILE.exists():
    print(f"ERROR: Prompt file not found: {PROMPT_FILE}")
    sys.exit(1)
SYSTEM_PROMPT = PROMPT_FILE.read_text(encoding="utf-8").strip()

MAX_WORKERS_PER_MODEL = 5
MAX_RETRIES = 3
MAX_REQUESTS_PER_MINUTE = 400


class RateLimiter:
    """Thread-safe token bucket rate limiter."""

    def __init__(self, max_per_minute: int):
        self._interval = 60.0 / max_per_minute
        self._lock = threading.Lock()
        self._last_call = 0.0

    def acquire(self):
        with self._lock:
            now = time.time()
            wait = self._last_call + self._interval - now
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.time()


rate_limiter = RateLimiter(MAX_REQUESTS_PER_MINUTE)


def evaluate_single(question: str, expected_answer: str, llm_answer: str) -> int:
    """Call OpenAI to score a single answer. Returns 0 or 1."""
    client = OpenAI(api_key=API_KEY)
    user_text = (
        f'{{\n"Question": "{question}",\n"Expected Answer": "{expected_answer}",\n'
        f'"LLM Answer": "\\n{llm_answer.strip()}"\n}}'
    )
    messages = [
        {
            "role": "developer",
            "content": [{"type": "input_text", "text": SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [{"type": "input_text", "text": user_text}],
        },
    ]

    for attempt in range(MAX_RETRIES):
        try:
            rate_limiter.acquire()
            response = client.responses.create(
                model="gpt-5.1",
                input=messages,
                text={"format": {"type": "text"}, "verbosity": "medium"},
                reasoning={"effort": "medium", "summary": "auto"},
                tools=[],
                store=True,
                include=["reasoning.encrypted_content", "web_search_call.action.sources"],
            )
            result_text = response.output[1].content[0].text
            result_text = result_text.replace("\n", "").replace("'", "")
            return json.loads(result_text)["score"]
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"  WARNING: Failed to score after {MAX_RETRIES} attempts: {e}")
                return -1


def score_column(df: pd.DataFrame, column: str) -> list:
    """Score all rows in a given column using parallel threads."""
    scores = [None] * len(df)

    def _score_row(idx):
        row = df.iloc[idx]
        return idx, evaluate_single(row["Question"], row["Answer"], str(row[column]))

    with ThreadPoolExecutor(max_workers=MAX_WORKERS_PER_MODEL) as executor:
        futures = {executor.submit(_score_row, i): i for i in range(len(df))}
        completed = 0
        for future in as_completed(futures):
            idx, score = future.result()
            scores[idx] = score
            completed += 1
            if completed % 50 == 0:
                print(f"    Scored {completed}/{len(df)} rows...")

    return scores


def compute_statistics(scores: list) -> dict:
    """Compute mean, std, median, and 95% confidence interval for a list of scores."""
    valid = [s for s in scores if s >= 0]
    if not valid:
        return {"mean": None, "std": None, "median": None, "ci_95_lower": None, "ci_95_upper": None, "n": 0}

    arr = np.array(valid, dtype=float)
    n = len(arr)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    median = float(np.median(arr))

    if n > 1:
        se = std / np.sqrt(n)
        t_crit = stats.t.ppf(0.975, df=n - 1)
        ci_lower = mean - t_crit * se
        ci_upper = mean + t_crit * se
    else:
        ci_lower = mean
        ci_upper = mean

    return {
        "mean": round(mean, 4),
        "std": round(std, 4),
        "median": round(median, 4),
        "ci_95_lower": round(ci_lower, 4),
        "ci_95_upper": round(ci_upper, 4),
        "n": n,
    }


# Define the files and which columns to score in each
FILES_TO_SCORE = [
    {
        "path": OUTPUT_DIR / "ValidationDataset_MiniCPM-1B-sft-bf16_ToxicSuppressed.jsonl",
        "columns": ["OOB", "OOB_ToxicSuppressed"],
        "output_name": "MiniCPM-1B-sft-bf16_ToxicSuppressed_scored.jsonl",
    },
    {
        "path": OUTPUT_DIR / "ValidationDataset_MiniCPM-1B-sft-bf16-LoRA_onlyGoodExamples.jsonl",
        "columns": ["FT", "OOB"],
        "output_name": "MiniCPM-1B-sft-bf16-LoRA_onlyGoodExamples_scored.jsonl",
    },
    {
        "path": OUTPUT_DIR / "ValidationDataset_MiniCPM-1B-sft-bf16-LoRA_onlyRejectingQuestions.jsonl",
        "columns": ["FT", "OOB"],
        "output_name": "MiniCPM-1B-sft-bf16-LoRA_onlyRejectingQuestions_scored.jsonl",
    },
]


def score_file(file_config: dict) -> dict:
    """Score a single file for its specified columns."""
    file_path = file_config["path"]
    columns = file_config["columns"]
    output_name = file_config["output_name"]

    print(f"\n{'='*60}")
    print(f"Scoring: {file_path.name}")
    print(f"  Columns: {columns}")
    print(f"{'='*60}")

    df = pd.read_json(file_path, lines=True)
    print(f"  Loaded {len(df)} rows")

    all_stats = {}
    for col in columns:
        print(f"  Scoring {col} responses...")
        scores = score_column(df, col)
        df[f"{col}_Score"] = scores
        all_stats[col] = compute_statistics(scores)

    scored_path = OUTPUT_DIR / output_name
    df.to_json(scored_path, orient="records", lines=True)
    print(f"  Saved scored results to: {scored_path.name}")

    return {
        "file": file_path.name,
        "stats": all_stats,
    }


def main():
    print("=" * 60)
    print("  ABLATION ANALYSIS SCORING PIPELINE")
    print("=" * 60)

    # Verify all input files exist
    for fc in FILES_TO_SCORE:
        if not fc["path"].exists():
            print(f"ERROR: File not found: {fc['path']}")
            sys.exit(1)

    print(f"\nFiles to score: {len(FILES_TO_SCORE)}")
    for fc in FILES_TO_SCORE:
        print(f"  - {fc['path'].name} -> columns: {fc['columns']}")

    results = []
    for fc in FILES_TO_SCORE:
        result = score_file(fc)
        results.append(result)

    # Save summary results
    output_json = OUTPUT_DIR / "ablation_scoring_results.json"
    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSummary saved to: {output_json}")

    # Display scorecard
    print("\n")
    print("=" * 90)
    print(f"{'ABLATION SCORECARD':^90}")
    print("=" * 90)
    print(f"{'File':<55} {'Column':<20} {'Mean':<8} {'Std':<8} {'95% CI':<20} {'N':<5}")
    print("-" * 90)

    for r in results:
        file_name = r["file"]
        for col, st in r["stats"].items():
            if st["mean"] is not None:
                ci_str = f"[{st['ci_95_lower']:.4f}, {st['ci_95_upper']:.4f}]"
                print(f"{file_name:<55} {col:<20} {st['mean']:<8.4f} {st['std']:<8.4f} {ci_str:<20} {st['n']:<5}")
            else:
                print(f"{file_name:<55} {col:<20} {'N/A':<8} {'N/A':<8} {'N/A':<20} {0:<5}")
        print("-" * 90)

    print("\nDone!")


if __name__ == "__main__":
    main()
