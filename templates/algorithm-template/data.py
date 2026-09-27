"""Dataset loading.

Datasets live in the repo-level data/ directory and must be registered in
data/README.md. Each algorithm writes its own loader — usually under 20
lines; extract shared code only after the same logic has been copied 3+
times (the "rule of three").
"""

from paths import RAW_DATA_DIR


def load_data():
    """Load the dataset used by this algorithm.

    TODO: read from RAW_DATA_DIR / "<dataset>" (see data/README.md).
    """
    raise NotImplementedError(f"register and load data under {RAW_DATA_DIR}")
