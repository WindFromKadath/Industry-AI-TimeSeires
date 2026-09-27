"""Dataset loading.

Default dataset `synthetic-point-anomaly` is generated in memory (no files
needed). Real datasets follow the repo-wide standard format (see
data/README.md): data/processed/<name>/{train.csv,test.csv,meta.json}.
"""

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from paths import PROCESSED_DATA_DIR

SYNTHETIC_NAME = "synthetic-point-anomaly"


@dataclass(frozen=True)
class Dataset:
    """Univariate series split into train/test, with point-anomaly labels
    (1 = anomaly) available for the test part only."""

    name: str
    train: NDArray[np.floating]
    test: NDArray[np.floating]
    test_labels: NDArray[np.int_]


def load_dataset(name: str, seed: int = 42) -> Dataset:
    """Load a registered dataset by its name in data/README.md."""
    if name == SYNTHETIC_NAME:
        return _make_synthetic(seed)
    ds_dir = PROCESSED_DATA_DIR / name
    if not ds_dir.is_dir():
        raise FileNotFoundError(
            f"dataset '{name}' not found at {ds_dir}; register it in "
            "data/README.md and run scripts/preprocess_<name>.py first"
        )
    meta = json.loads((ds_dir / "meta.json").read_text(encoding="utf-8"))
    train = pd.read_csv(ds_dir / "train.csv")[meta["value_column"]].to_numpy()
    test_df = pd.read_csv(ds_dir / "test.csv")
    return Dataset(
        name=name,
        train=train,
        test=test_df[meta["value_column"]].to_numpy(),
        test_labels=test_df[meta["label_column"]].to_numpy(),
    )


def _make_synthetic(
    seed: int, n_samples: int = 4096, anomaly_ratio: float = 0.03
) -> Dataset:
    """Trend + seasonality + noise with random spikes; last 30% is test."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples)
    values = (
        0.01 * t
        + 10.0 * np.sin(2.0 * np.pi * t / 144.0)
        + rng.normal(0.0, 0.5, n_samples)
    )
    labels = np.zeros(n_samples, dtype=int)
    n_anomalies = int(n_samples * anomaly_ratio)
    idx = rng.choice(n_samples, size=n_anomalies, replace=False)
    spike = rng.choice([-1.0, 1.0], size=n_anomalies)
    values[idx] += spike * rng.uniform(5.0, 10.0, n_anomalies)
    labels[idx] = 1
    split = int(n_samples * 0.7)
    return Dataset(
        name=SYNTHETIC_NAME,
        train=values[:split],
        test=values[split:],
        test_labels=labels[split:],
    )


def make_features(values: NDArray[np.floating], window: int) -> NDArray[np.floating]:
    """Per-point features from a centered rolling window.

    Centering lets the window mean cancel trend and seasonality, so the
    residual (center value minus window mean) exposes spikes. Features are
    [residual, window std]. Points nearer than window // 2 to the series
    edges get no features; feature row 0 aligns with label index
    window // 2.
    """
    windows = np.lib.stride_tricks.sliding_window_view(values, window)
    center = windows[:, window // 2]
    return np.column_stack([center - windows.mean(axis=1), windows.std(axis=1)])
