"""
UAT Cleanup Script
==================
Walks each subdirectory under the UAT folder and removes everything
EXCEPT folders named 'completions', 'results', or 'test_cases'
(and the files inside those protected folders).

Usage:
    python cleanup_uat.py            # dry-run (preview only, nothing deleted)
    python cleanup_uat.py --execute  # actually delete
"""

import os
import shutil
import argparse

# ── Configuration ──────────────────────────────────────────────────────────────
UAT_ROOT       = os.path.dirname(os.path.abspath(__file__))   # same dir as this script
PROTECTED_DIRS = {"completions", "results", "test_cases"}
# ──────────────────────────────────────────────────────────────────────────────


def cleanup(dry_run: bool = True):
    mode_label = "[DRY RUN]" if dry_run else "[DELETING]"

    if not os.path.isdir(UAT_ROOT):
        print(f"ERROR: UAT root not found: {UAT_ROOT}")
        return

    # Iterate over direct children of UAT_ROOT (the subdirectories)
    for entry in sorted(os.scandir(UAT_ROOT), key=lambda e: e.name):
        # Skip the script itself (top-level file)
        if entry.is_file():
            continue

        subdir = entry.path
        print(f"\nProcessing: {subdir}")

        # Iterate over items inside this subdirectory
        for item in sorted(os.scandir(subdir), key=lambda e: e.name):
            # Protected folder → skip entirely (keep folder + contents)
            if item.is_dir() and item.name.lower() in PROTECTED_DIRS:
                print(f"  KEEP  (protected)  {item.name}/")
                continue

            # Everything else → delete
            if item.is_dir():
                print(f"  {mode_label}  DIR   {item.name}/")
                if not dry_run:
                    shutil.rmtree(item.path)
            else:
                print(f"  {mode_label}  FILE  {item.name}")
                if not dry_run:
                    os.remove(item.path)

    if dry_run:
        print("\n── Dry-run complete. Run with --execute to apply deletions. ──")
    else:
        print("\n── Cleanup complete. ──")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UAT folder cleanup utility")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually delete files/folders (default is dry-run / preview only)",
    )
    args = parser.parse_args()
    cleanup(dry_run=not args.execute)
