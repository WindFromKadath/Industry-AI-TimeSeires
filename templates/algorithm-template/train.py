"""Training / evaluation entry point.

Run from this directory:
    uv run python train.py
"""

import json

from config import Config
from paths import OUTPUT_DIR


def main() -> None:
    config = Config()
    print(f"running with seed={config.seed}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # TODO: load data -> fit model -> evaluate
    metrics: dict[str, float] = {}
    out_file = OUTPUT_DIR / "metrics.json"
    out_file.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"metrics written to {out_file}")


if __name__ == "__main__":
    main()
