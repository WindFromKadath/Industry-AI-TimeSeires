"""Result visualization — the human-readable counterpart of metrics.json."""

import matplotlib.pyplot as plt
from numpy.typing import NDArray


def plot_scores(
    test_values: NDArray,
    test_labels: NDArray,
    scores: NDArray,
    window: int,
    out_png,
) -> None:
    """Two stacked panels sharing the time axis: the test series with true
    anomalies marked, and the model's anomaly score for the same points."""
    offset = window // 2
    score_index = range(offset, offset + len(scores))
    anomaly_index = [i for i, y in enumerate(test_labels) if y == 1]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=False)
    ax1.plot(range(len(test_values)), test_values, lw=0.8)
    ax1.scatter(
        anomaly_index, test_values[anomaly_index],
        c="red", s=12, zorder=3, label="true anomaly",
    )
    ax1.legend(loc="upper right")
    ax1.set_ylabel("value")
    ax1.set_title("test series with true anomalies")
    ax2.plot(score_index, scores, lw=0.8, c="darkorange")
    ax2.set_ylabel("anomaly score")
    ax2.set_xlabel("time index")
    ax2.set_title("isolation forest score (higher = more anomalous)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
