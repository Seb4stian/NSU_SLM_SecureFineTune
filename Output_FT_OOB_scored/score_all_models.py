"""
Score all consolidated model outputs (FT and OOB) in parallel using OpenAI GPT-5.1.
Produces a scorecard with statistics and saves results to JSON.
"""

import os
import json
import glob
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

# Load API key from .env file in this directory
load_dotenv(Path(__file__).parent / ".env")
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY or API_KEY == "your-api-key-here":
    print("ERROR: Please set your OpenAI API key in the .env file")
    sys.exit(1)

CONSOLIDATED_DIR = Path(__file__).parent / "consolidated_answers"
OUTPUT_DIR = Path(__file__).parent
PROMPTS_DIR = Path(__file__).parent / "prompts"
MAX_WORKERS_PER_MODEL = 5  # concurrent API calls per model
MAX_RETRIES = 3
MAX_REQUESTS_PER_MINUTE = 400  # global rate limit (adjust based on your OpenAI tier)


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


# Global rate limiter shared across all threads
rate_limiter = RateLimiter(MAX_REQUESTS_PER_MINUTE)

# Load system prompt from external file
PROMPT_FILE = PROMPTS_DIR / "scoring_system_prompt.txt"
if not PROMPT_FILE.exists():
    print(f"ERROR: Prompt file not found: {PROMPT_FILE}")
    sys.exit(1)
SYSTEM_PROMPT = PROMPT_FILE.read_text(encoding="utf-8").strip()


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
                return -1  # mark as failed


def score_column(df: pd.DataFrame, column: str) -> list:
    """Score all rows in a given column (FT or OOB) using parallel threads."""
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

    # 95% confidence interval using t-distribution
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


def score_model(file_path: Path) -> dict:
    """Score a single model file for both FT and OOB columns."""
    model_name = file_path.stem.replace("consolidated_", "")
    print(f"\n{'='*60}")
    print(f"Scoring model: {model_name}")
    print(f"{'='*60}")

    df = pd.read_json(file_path, lines=True)
    print(f"  Loaded {len(df)} rows")

    # Score FT
    print(f"  Scoring FT responses...")
    ft_scores = score_column(df, "FT")
    df["FT_Score"] = ft_scores

    # Score OOB
    print(f"  Scoring OOB responses...")
    oob_scores = score_column(df, "OOB")
    df["OOB_Score"] = oob_scores

    # Save scored file
    scored_path = OUTPUT_DIR / f"{model_name}_scored.jsonl"
    df.to_json(scored_path, orient="records", lines=True)
    print(f"  Saved scored results to: {scored_path.name}")

    ft_stats = compute_statistics(ft_scores)
    oob_stats = compute_statistics(oob_scores)

    return {
        "model": model_name,
        "FT": ft_stats,
        "OOB": oob_stats,
    }


def main():
    print("=" * 60)
    print("  MODEL SCORING PIPELINE")
    print("=" * 60)

    # Find all consolidated answer files
    files = sorted(CONSOLIDATED_DIR.glob("consolidated_*.jsonl"))
    if not files:
        print(f"ERROR: No consolidated_*.jsonl files found in {CONSOLIDATED_DIR}")
        sys.exit(1)

    print(f"Found {len(files)} model files to score:")
    for f in files:
        print(f"  - {f.name}")

    # Score all models in parallel (models concurrent + rows concurrent within each)
    results = []
    with ThreadPoolExecutor(max_workers=len(files)) as model_executor:
        future_to_file = {model_executor.submit(score_model, f): f for f in files}
        for future in as_completed(future_to_file):
            results.append(future.result())

    # Sort results by model name for consistent output
    results.sort(key=lambda r: r["model"])

    # Save results to JSON
    output_json = OUTPUT_DIR / "scoring_results.json"
    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_json}")

    # Display scorecard
    print("\n")
    print("=" * 80)
    print(f"{'SCORECARD':^80}")
    print("=" * 80)
    print(f"{'Model':<35} {'Type':<5} {'Mean':<8} {'Std':<8} {'Median':<8} {'95% CI':<20} {'N':<5}")
    print("-" * 80)

    for r in results:
        model = r["model"]
        for label, st in [("FT", r["FT"]), ("OOB", r["OOB"])]:
            if st["mean"] is not None:
                ci_str = f"[{st['ci_95_lower']:.4f}, {st['ci_95_upper']:.4f}]"
                print(f"{model:<35} {label:<5} {st['mean']:<8.4f} {st['std']:<8.4f} {st['median']:<8.4f} {ci_str:<20} {st['n']:<5}")
            else:
                print(f"{model:<35} {label:<5} {'N/A':<8} {'N/A':<8} {'N/A':<8} {'N/A':<20} {0:<5}")
        print("-" * 80)

    print("\nDone!")


if __name__ == "__main__":
    main()
