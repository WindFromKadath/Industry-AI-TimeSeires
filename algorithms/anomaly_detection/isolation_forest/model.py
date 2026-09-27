"""Isolation Forest for point-anomaly detection in time series.

Paper mapping (Liu et al., ICDM 2008, Section 3): the anomaly score derives
from the average path length over an ensemble of isolation trees (Eq. 1-3);
points that isolate in fewer splits are more likely anomalous.
"""

from sklearn.ensemble import IsolationForest


def build_model(n_estimators: int, contamination: float, seed: int) -> IsolationForest:
    """Create the Isolation Forest ensemble (paper Section 3.1-3.2)."""
    return IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=seed,
    )
