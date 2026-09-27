"""Training / evaluation entry point.

Run from this directory:
    uv run python train.py --dataset <name>
    uv run python train.py --dataset <name> --seed 0

Every run writes one record to outputs/{{ALGO_NAME}}/<dataset>/metrics.json
so results stay comparable across datasets and algorithms.
"""

import argparse
import json
from datetime import datetime, timezone

from config import Config
from paths import OUTPUT_DIR

from data import load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", required=True, help="registered dataset name (data/README.md)"
    )
    parser.add_argument("--seed", type=int, default=Config.seed)
    # TODO: expose algorithm hyperparameters here, defaulting to Config fields
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = Config(seed=args.seed)
    print(f"running with seed={config.seed}")
    dataset = load_dataset(args.dataset)
    # TODO: fit on dataset.train, evaluate on dataset.test
    metrics: dict[str, float] = {}
    record = {
        "dataset": dataset.name,
        "params": vars(args),
        "metrics": metrics,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    out_dir = OUTPUT_DIR / dataset.name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "metrics.json"
    out_file.write_text(
        json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"metrics written to {out_file}; refresh docs/benchmarks.md with:")
    print("    uv run python scripts/collect_results.py")


if __name__ == "__main__":
    main()
