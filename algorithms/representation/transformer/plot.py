"""Training-curve visualization — the human-readable counterpart of metrics.json.

Deliberate copy of templates/algorithm-template/plot.py: every algorithm
owns its plotting code (see AGENTS.md, decoupling rules).
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
