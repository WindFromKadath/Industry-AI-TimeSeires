"""Dataset loading via the repo-wide standard format.

Standard layout for every registered dataset (see data/README.md):
    data/processed/<name>/train.csv
    data/processed/<name>/test.csv
    data/processed/<name>/meta.json   # task, value_column, label_column, ...

Swapping datasets must never require code changes here — only a different
`--dataset` name. Keep this loader self-contained; extract shared code only
after the same logic has been copied 3+ times (the "rule of three").
"""

import json
from dataclasses import dataclass

import pandas as pd
from paths import PROCESSED_DATA_DIR


@dataclass(frozen=True)
class Dataset:
    name: str
    train: pd.DataFrame
    test: pd.DataFrame
    meta: dict


def load_dataset(name: str) -> Dataset:
    """Load a registered dataset by its name in data/README.md."""
    ds_dir = PROCESSED_DATA_DIR / name
    if not ds_dir.is_dir():
        raise FileNotFoundError(
            f"dataset '{name}' not found at {ds_dir}; register it in "
            "data/README.md and run scripts/preprocess_<name>.py first"
        )
    meta = json.loads((ds_dir / "meta.json").read_text(encoding="utf-8"))
    return Dataset(
        name=name,
        train=pd.read_csv(ds_dir / "train.csv"),
        test=pd.read_csv(ds_dir / "test.csv"),
        meta=meta,
    )
