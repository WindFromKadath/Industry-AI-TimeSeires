"""Smoke test: the full pipeline runs on a small synthetic series."""

from config import Config
from model import build_model

from data import make_features, make_series


def test_pipeline_smoke():
    config = Config(n_samples=512, n_estimators=50)
    values, labels = make_series(
        config.n_samples, config.anomaly_ratio, config.seed
    )
    features = make_features(values, config.window)
    model = build_model(config.n_estimators, config.contamination, config.seed)
    model.fit(features)
    scores = -model.decision_function(features)
    assert len(scores) == config.n_samples - config.window + 1
    assert labels.sum() > 0
