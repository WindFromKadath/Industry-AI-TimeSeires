"""Smoke test: core objects can be constructed with default config."""

from config import Config


def test_config_defaults():
    assert Config().seed == 42
