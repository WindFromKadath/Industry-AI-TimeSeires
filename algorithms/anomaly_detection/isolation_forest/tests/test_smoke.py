"""Smoke tests: synthetic pipeline runs; missing real datasets fail loudly."""

import pytest
from model import build_model

from data import load_dataset, make_features


def test_pipeline_smoke():
    dataset = load_dataset("synthetic-point-anomaly", seed=0)
    window = 24
    train_feats = make_features(dataset.train, window)
    test_feats = make_features(dataset.test, window)
    model = build_model(n_estimators=50, contamination=0.03, seed=0)
    model.fit(train_feats)
    scores = -model.decision_function(test_feats)
    assert len(scores) == len(dataset.test) - window + 1
    assert dataset.test_labels.sum() > 0


def test_missing_dataset_raises():
    with pytest.raises(FileNotFoundError, match="register it in"):
        load_dataset("dataset-that-does-not-exist")
