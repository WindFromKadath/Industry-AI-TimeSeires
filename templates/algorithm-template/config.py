"""Hyperparameters and experiment configuration."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    seed: int = 42
    # TODO: add algorithm hyperparameters here
