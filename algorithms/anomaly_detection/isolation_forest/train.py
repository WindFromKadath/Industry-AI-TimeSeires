"""Train + evaluate Isolation Forest on the synthetic anomaly dataset.

Run from this directory:
    uv run python train.py
"""

import json

from config import Config
from model import build_model
from paths import OUTPUT_DIR
from sklearn.metrics import average_precision_score, roc_auc_score

from data import make_features, make_series


def main() -> None:
    config = Config()
    values, labels = make_series(
        config.n_samples, config.anomaly_ratio, config.seed
    )
    features = make_features(values, config.window)
    offset = config.window // 2
    window_labels = labels[offset : offset + len(features)]

    model = build_model(config.n_estimators, config.contamination, config.seed)
    model.fit(features)
    # sklearn decision_function: higher = more normal -> negate to score
    scores = -model.decision_function(features)

    metrics = {
        "average_precision": round(float(average_precision_score(window_labels, scores)), 4),
        "roc_auc": round(float(roc_auc_score(window_labels, scores)), 4),
        "n_samples": config.n_samples,
        "window": config.window,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUTPUT_DIR / "metrics.json"
    out_file.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"metrics written to {out_file}")


if __name__ == "__main__":
    main()
