"""Preprocess WikiText-2 (raw) into the repo-wide standard format.

Reads data/raw/wikitext-2-raw/wikitext-2-raw/wiki.{train,test}.raw and writes
data/processed/wikitext-2-raw/{train.csv,test.csv,meta.json}:
whitespace-tokenized word ids (top-V vocabulary from the train split; ids
0=PAD, 1=BOS, 2=UNK are reserved), chunked into non-overlapping fixed-length
windows for the continuation objective — the first `context_len` tokens of a
window are the encoder context, the rest is the decoder target.

Run from repo root:
    uv run python scripts/preprocess_wikitext_2_raw.py
"""

import json
from collections import Counter
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw" / "wikitext-2-raw" / "wikitext-2-raw"
OUT_DIR = REPO_ROOT / "data" / "processed" / "wikitext-2-raw"

VOCAB_SIZE = 16000
WINDOW_LEN = 64
CONTEXT_LEN = 32
MAX_TRAIN_WINDOWS = 20000
MAX_TEST_WINDOWS = 2000
UNK_ID = 2


def tokenize(path: Path) -> list[str]:
    """Whitespace tokens, skipping heading lines (' = Title = ') and blanks."""
    tokens = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or (line.startswith("=") and line.endswith("=")):
            continue
        tokens.extend(line.split())
    return tokens


def to_windows(ids: list[int], window_len: int, max_windows: int) -> list[list[int]]:
    windows = [
        ids[i : i + window_len]
        for i in range(0, len(ids) - window_len + 1, window_len)
    ]
    return windows[:max_windows]


def main() -> None:
    train_tokens = tokenize(RAW_DIR / "wiki.train.raw")
    test_tokens = tokenize(RAW_DIR / "wiki.test.raw")

    counts = Counter(train_tokens)
    vocab = {tok: idx + 3 for idx, (tok, _) in enumerate(counts.most_common(VOCAB_SIZE - 3))}

    train_windows = to_windows(
        [vocab.get(tok, UNK_ID) for tok in train_tokens], WINDOW_LEN, MAX_TRAIN_WINDOWS
    )
    test_windows = to_windows(
        [vocab.get(tok, UNK_ID) for tok in test_tokens], WINDOW_LEN, MAX_TEST_WINDOWS
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for split, windows in [("train", train_windows), ("test", test_windows)]:
        df = pd.DataFrame({"tokens": [" ".join(map(str, w)) for w in windows]})
        df.to_csv(OUT_DIR / f"{split}.csv", index=False)

    meta = {
        "task": "representation",
        "value_column": "tokens",
        "vocab_size": VOCAB_SIZE,
        "objective": "continuation",
        "context_len": CONTEXT_LEN,
        "preprocessing": (
            "scripts/preprocess_wikitext_2_raw.py: 空白分词并去除标题/空行;"
            f"词表取训练集 top {VOCAB_SIZE - 3}(0=PAD,1=BOS,2=UNK);"
            f"token 流切分为不重叠窗口(长度 {WINDOW_LEN})"
        ),
        "split": (
            "官方划分:wiki.train.raw -> train.csv"
            f"(前 {MAX_TRAIN_WINDOWS} 窗口),wiki.test.raw -> test.csv"
            f"(前 {MAX_TEST_WINDOWS} 窗口)"
        ),
        "source": (
            "WikiText-2 raw v1 (Merity et al., 2016), "
            "https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/"
        ),
    }
    (OUT_DIR / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"train: {len(train_tokens)} tokens -> {len(train_windows)} windows")
    print(f"test:  {len(test_tokens)} tokens -> {len(test_windows)} windows")
    print(f"written to {OUT_DIR.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
