"""Repo-level path constants for this algorithm.

Deliberately kept tiny and self-contained: every algorithm owns its own copy
(see CONTRIBUTING.md, the "rule of three") so deleting one algorithm
directory never breaks another.
"""

from pathlib import Path

# algorithms/<category>/{{ALGO_NAME}}/paths.py -> parents[3] is the repo root
REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUT_DIR = REPO_ROOT / "outputs" / "{{ALGO_NAME}}"
# train.py writes one run record to OUTPUT_DIR / <dataset> / metrics.json;
# scripts/collect_results.py aggregates all of them into docs/benchmarks.md
