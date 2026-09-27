"""Synthetic univariate time series with injected point anomalies.

Self-contained (no external dataset) so the skeleton can be verified
end-to-end; see data/README.md for how real datasets are registered.
"""

import numpy as np
from numpy.typing import NDArray


def make_series(
    n_samples: int, anomaly_ratio: float, seed: int
) -> tuple[NDArray[np.floating], NDArray[np.int_]]:
    """Trend + seasonality + noise, with random large spikes as anomalies.

    Returns:
        values: shape (n_samples,), the observed series.
        labels: shape (n_samples,), 1 = anomaly, 0 = normal.
    """
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
    return values, labels


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
