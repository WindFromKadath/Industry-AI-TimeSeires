"""Dataset loading.

Default dataset `synthetic-copy-task` is generated in memory (no files
needed): random fixed-length symbol sequences the model must reproduce —
the classic sanity task for encoder-decoder architectures. Real
token-sequence datasets follow the repo-wide standard format (see
data/README.md): data/processed/<name>/{train.csv,test.csv,meta.json},
where meta["value_column"] names a column of space-separated token ids and
meta["vocab_size"] gives the vocabulary size.

Each sequence is turned into an (src, tgt) pair according to
meta["objective"]: "copy" sets tgt = src; "continuation" splits the sequence
at meta["context_len"] so the decoder continues what the encoder reads.
Swapping datasets must never require code changes here — only a different
`--dataset` name. Keep this loader self-contained; extract shared code only
after the same logic has been copied 3+ times (the "rule of three").
"""

import json
from dataclasses import dataclass

import pandas as pd
from paths import PROCESSED_DATA_DIR

SYNTHETIC_NAME = "synthetic-copy-task"
PAD_ID = 0
BOS_ID = 1


@dataclass(frozen=True)
class Dataset:
    """Token-id sequences for an encoder-decoder task.

    Sequences hold symbol ids in [2, vocab_size); PAD_ID/BOS_ID are reserved
    and prepended by the training loop, not stored here. `objective` decides
    how a sequence becomes an (src, tgt) pair; see the module docstring.
    """

    name: str
    train: list[list[int]]
    test: list[list[int]]
    vocab_size: int
    objective: str = "copy"
    context_len: int = 0


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
    column = meta["value_column"]
    return Dataset(
        name=name,
        train=_parse_id_column(pd.read_csv(ds_dir / "train.csv"), column),
        test=_parse_id_column(pd.read_csv(ds_dir / "test.csv"), column),
        vocab_size=int(meta["vocab_size"]),
        objective=meta.get("objective", "copy"),
        context_len=int(meta.get("context_len", 0)),
    )


def _parse_id_column(df: pd.DataFrame, column: str) -> list[list[int]]:
    return [[int(tok) for tok in row.split()] for row in df[column]]


def _make_synthetic(
    seed: int,
    n_train: int = 9600,
    n_test: int = 1024,
    seq_len: int = 10,
    vocab_size: int = 16,
) -> Dataset:
    """Random symbol sequences over [2, vocab_size); the target is the copy."""
    import numpy as np

    rng = np.random.default_rng(seed)
    train = rng.integers(2, vocab_size, size=(n_train, seq_len)).tolist()
    test = rng.integers(2, vocab_size, size=(n_test, seq_len)).tolist()
    return Dataset(
        name=SYNTHETIC_NAME,
        train=train,
        test=test,
        vocab_size=vocab_size,
    )
