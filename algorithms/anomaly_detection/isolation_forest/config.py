"""Hyperparameters and experiment configuration."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    seed: int = 42
    n_samples: int = 4096
    anomaly_ratio: float = 0.03
    window: int = 24
    n_estimators: int = 200
    contamination: float = 0.03
