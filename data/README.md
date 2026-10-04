# Dataset Registry

[中文](README.zh-CN.md) | English

All datasets are stored under this directory. **Data files are not committed to Git** (see `.gitignore`); this table is the only content that needs committing.

## Directory conventions

- `raw/<dataset>/` — raw data; append-only, kept as the source provided it
- `processed/<dataset>/` — preprocessed artifacts; must follow the **standard format** below; preprocessing scripts live in `scripts/preprocess_<dataset>.py`
- Algorithms only **read** `data/`; run results go to `outputs/<algo>/<dataset>/`; no data inside algorithm directories

## Standard format for preprocessed artifacts

An algorithm's `load_dataset(name)` depends only on "registered name + this format"; switching datasets does not change code:

```
data/processed/<dataset>/
├── train.csv     # Training segment (for anomaly detection may be normal-only)
├── test.csv      # Test segment (with label column)
└── meta.json     # See below
```

`meta.json` fields:

| Field | Meaning |
|---|---|
| `task` | anomaly_detection / forecasting / classification / representation |
| `value_column` | Value column name (list of column names when multivariate) |
| `label_column` | Label column name (may be omitted for unlabeled tasks) |
| `vocab_size` | Vocabulary size (for token-id sequence tasks) |
| `objective` | Training objective (for token sequence tasks): copy = copy, continuation = continuation (default copy) |
| `context_len` | Under the continuation objective, the first context_len tokens are the encoder context |
| `preprocessing` | Preprocessing notes (scaling, detrending, etc.) and the corresponding script path |
| `split` | Description of the split method |
| `source` | Provenance of the raw data |

## Datasets

| Registered name | Source / link | License | Format and fields | Applicable tasks | Preprocessing and split |
|---|---|---|---|---|---|
| synthetic-point-anomaly | In-memory synthesis, no files (see `algorithms/anomaly_detection/isolation_forest/data.py`) | — | Univariate time series + 0/1 point-anomaly labels | Anomaly detection | Centered sliding-window (24) features; first 70% train / last 30% test |
| synthetic-copy-task | In-memory synthesis, no files (see `algorithms/representation/transformer/data.py`) | — | Fixed-length random symbol sequences (0=PAD, 1=BOS, 2..15 symbols, vocab 16); target = copy of source sequence | Representation (sequence-transduction smoke test) | 9600 train / 1024 test, sequence length 10 |
| wikitext-2-raw | [WikiText-2 raw v1 (Merity et al., 2016)](https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/); raw files in `data/raw/wikitext-2-raw/` (downloaded via ModelScope mirror) | CC BY-SA 4.0 | Word-level English text; `scripts/preprocess_wikitext_2_raw.py` produces fixed-length 64-token windows (first 32 = encoder context, last 32 = continuation target), vocab 16000 (0=PAD, 1=BOS, 2=UNK) | Representation (sequence continuation) | Whitespace tokenization, headings/blank lines removed, top-V vocabulary from the training set, non-overlapping windowing of the token stream; official train → first 20000 windows, test → first 2000 windows |

When adding a dataset, register a row in this table; update download links promptly when they break.
