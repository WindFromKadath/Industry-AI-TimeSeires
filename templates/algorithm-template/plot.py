"""Training-curve visualization — the human-readable training artifact.

Convention (see CONTRIBUTING.md): every run must produce, besides
metrics.json, at least one picture a human can read at a glance. Iterative
algorithms record per-epoch history to history.csv during training and
render it here; non-iterative algorithms instead plot their result (e.g.
anomaly scores over the series — see the isolation_forest example).
"""

import matplotlib.pyplot as plt
import pandas as pd


def plot_history(history_csv, out_png) -> None:
    """Plot every numeric column except `epoch` against epoch."""
    history = pd.read_csv(history_csv)
    fig, ax = plt.subplots(figsize=(8, 5))
    for column in history.columns:
        if column != "epoch":
            ax.plot(history["epoch"], history[column], label=column)
    ax.set_xlabel("epoch")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
