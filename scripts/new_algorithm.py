"""Scaffold a new algorithm directory from templates/algorithm-template.

Usage (from anywhere in the repo):
    uv run scripts/new_algorithm.py <category> <algo_name>

Example:
    uv run scripts/new_algorithm.py anomaly_detection isolation_forest
"""

import argparse
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_DIR = REPO_ROOT / "templates" / "algorithm-template"
PLACEHOLDER = "{{ALGO_NAME}}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("category", help="task category, e.g. anomaly_detection")
    parser.add_argument("algo_name", help="algorithm directory, snake_case")
    args = parser.parse_args()

    if not args.algo_name.isidentifier() or args.algo_name != args.algo_name.lower():
        parser.error("algo_name must be a lowercase snake_case identifier")

    target = REPO_ROOT / "algorithms" / args.category / args.algo_name
    if target.exists():
        print(f"error: {target} already exists", file=sys.stderr)
        return 1

    shutil.copytree(TEMPLATE_DIR, target)
    for path in target.rglob("*"):
        if path.is_file():
            text = path.read_text(encoding="utf-8")
            if PLACEHOLDER in text:
                path.write_text(
                    text.replace(PLACEHOLDER, args.algo_name), encoding="utf-8"
                )

    rel = target.relative_to(REPO_ROOT)
    print(f"created {rel}")
    print("next steps:")
    print(f"  1. fill in {rel / 'README.md'} (paper link, dataset, status)")
    print("  2. register a row in the mapping table of docs/README.md")
    print(f"  3. cd {rel} && uv run python train.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
