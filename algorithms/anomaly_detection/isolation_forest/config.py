"""Hyperparameters and experiment configuration."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    seed: int = 42
    window: int = 24
    n_estimators: int = 200
    contamination: float = 0.03
