"""Run every algorithm's test suite in a separate subprocess, then summarize.

Algorithm directories intentionally reuse module names (model.py, data.py,
...), so their tests must not share one pytest process. Run from repo root:
    uv run python scripts/test_all.py
"""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    algo_dirs = sorted(
        d.parent for d in (REPO_ROOT / "algorithms").rglob("tests") if d.is_dir()
    )
    if not algo_dirs:
        print("no algorithm tests found")
        return 0

    failed = []
    for algo_dir in algo_dirs:
        print(f"== {algo_dir.relative_to(REPO_ROOT)}")
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-q"], cwd=algo_dir, check=False
        )
        if result.returncode != 0:
            failed.append(str(algo_dir.relative_to(REPO_ROOT)))

    if failed:
        print("FAILED: " + ", ".join(failed))
        return 1
    print(f"all {len(algo_dirs)} algorithm test suites passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
