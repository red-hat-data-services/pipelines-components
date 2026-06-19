"""Tests for time series integration test config JSON loading."""

from __future__ import annotations

from pathlib import Path

from . import test_configs

_TESTS_DIR = Path(__file__).resolve().parent


def test_all_dataset_paths_exist() -> None:
    """Every dataset_path in test_configs.json resolves to a file under tests/."""
    for config in test_configs.TEST_CONFIGS:
        dataset = _TESTS_DIR / config.dataset_path
        assert dataset.is_file(), f"Missing dataset for config {config.id!r}: {dataset}"


def test_eval_metric_forwarded_in_pipeline_arguments() -> None:
    """Configs with eval_metric pass it through get_pipeline_arguments()."""
    wql_config = next(c for c in test_configs.TEST_CONFIGS if c.id == "timeseries_wql_eval_metric")
    args = wql_config.get_pipeline_arguments("bucket", "key", "secret")
    assert args["eval_metric"] == "WQL"

    smoke_config = next(c for c in test_configs.TEST_CONFIGS if c.id == "timeseries_smoke")
    smoke_args = smoke_config.get_pipeline_arguments("bucket", "key", "secret")
    assert "eval_metric" not in smoke_args
