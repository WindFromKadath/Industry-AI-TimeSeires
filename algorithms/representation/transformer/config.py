"""Hyperparameters and experiment configuration.

Defaults are sized for the tiny synthetic copy task. The paper's base model
(Section 3: N=6, d_model=512, h=8, d_ff=2048; Section 5.3: warmup=4000) can
be reproduced via CLI flags on a real dataset.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    seed: int = 42
    # model architecture (paper Section 3)
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 512
    dropout: float = 0.1
    max_len: int = 64
    # training regime (paper Section 5)
    epochs: int = 30
    batch_size: int = 64
    warmup_steps: int = 400
    label_smoothing: float = 0.1
