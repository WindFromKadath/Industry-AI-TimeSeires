"""Train + evaluate Isolation Forest on a registered dataset.

Run from this directory:
    uv run python train.py                                  # synthetic default
    uv run python train.py --dataset synthetic-point-anomaly --seed 0

Every run writes one record to
outputs/isolation_forest/<dataset>/metrics.json so results stay comparable
across datasets and algorithms.
"""

import argparse
import json
from datetime import datetime, timezone

from config import Config
from model import build_model
from paths import OUTPUT_DIR
from sklearn.metrics import average_precision_score, roc_auc_score

from data import load_dataset, make_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="synthetic-point-anomaly")
    parser.add_argument("--seed", type=int, default=Config.seed)
    parser.add_argument("--n-estimators", type=int, default=Config.n_estimators)
    parser.add_argument("--contamination", type=float, default=Config.contamination)
    parser.add_argument("--window", type=int, default=Config.window)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = Config(
        seed=args.seed,
        n_estimators=args.n_estimators,
        contamination=args.contamination,
        window=args.window,
    )
    dataset = load_dataset(args.dataset, seed=config.seed)

    train_feats = make_features(dataset.train, config.window)
    test_feats = make_features(dataset.test, config.window)
    offset = config.window // 2
    test_labels = dataset.test_labels[offset : offset + len(test_feats)]

    model = build_model(config.n_estimators, config.contamination, config.seed)
    model.fit(train_feats)
    # sklearn decision_function: higher = more normal -> negate to score
    scores = -model.decision_function(test_feats)

    metrics = {
        "average_precision": round(float(average_precision_score(test_labels, scores)), 4),
        "roc_auc": round(float(roc_auc_score(test_labels, scores)), 4),
    }
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
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"metrics written to {out_file}; refresh docs/benchmarks.md with:")
    print("    uv run python scripts/collect_results.py")


if __name__ == "__main__":
    main()
