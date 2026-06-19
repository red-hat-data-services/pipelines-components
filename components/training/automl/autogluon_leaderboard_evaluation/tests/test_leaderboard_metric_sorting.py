"""Leaderboard sorting tests using real pandas (no DataFrame mocks)."""

import json
from pathlib import Path
from unittest import mock

import pytest

from ..component import leaderboard_evaluation


def _write_model_metrics(base_path: Path, model_name: str, metrics: dict) -> None:
    metrics_dir = base_path / model_name / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")


class TestLeaderboardMetricSorting:
    """Verify AutoGluon negated-metric convention produces correct best-model ranking."""

    def test_negated_rmse_ranks_higher_value_first(self, tmp_path):
        """Flipped RMSE (-0.3 beats -0.8) selects the better model as best_model."""
        combined_root = tmp_path / "models"
        _write_model_metrics(combined_root, "ModelA", {"root_mean_squared_error": -0.8})
        _write_model_metrics(combined_root, "ModelB", {"root_mean_squared_error": -0.3})

        models_artifact = mock.MagicMock()
        models_artifact.path = str(combined_root)
        models_artifact.uri = "http://example.com/artifacts"
        models_artifact.metadata = {"model_names": json.dumps(["ModelA", "ModelB"])}

        html_artifact = mock.MagicMock()
        html_artifact.path = str(tmp_path / "leaderboard.html")
        html_artifact.metadata = {}

        component_status = mock.MagicMock()
        component_status.path = str(tmp_path / "status")
        component_status.metadata = {}

        result = leaderboard_evaluation.python_func(
            models_artifact=models_artifact,
            eval_metric="root_mean_squared_error",
            html_artifact=html_artifact,
            component_status=component_status,
        )

        assert result.best_model == "ModelB"
        html = Path(html_artifact.path).read_text(encoding="utf-8")
        assert html.index("ModelB") < html.index("ModelA")

    def test_mase_ranks_higher_value_first(self, tmp_path):
        """Timeseries MASE values rank with higher-is-better AutoGluon convention."""
        combined_root = tmp_path / "models"
        _write_model_metrics(combined_root, "DeepAR", {"MASE": -0.55})
        _write_model_metrics(combined_root, "TFT", {"MASE": -0.21})

        models_artifact = mock.MagicMock()
        models_artifact.path = str(combined_root)
        models_artifact.uri = "http://example.com/artifacts"
        models_artifact.metadata = {"model_names": json.dumps(["DeepAR", "TFT"])}

        html_artifact = mock.MagicMock()
        html_artifact.path = str(tmp_path / "leaderboard_ts.html")
        html_artifact.metadata = {}

        component_status = mock.MagicMock()
        component_status.path = str(tmp_path / "status_ts")
        component_status.metadata = {}

        result = leaderboard_evaluation.python_func(
            models_artifact=models_artifact,
            eval_metric="MASE",
            html_artifact=html_artifact,
            component_status=component_status,
        )

        assert result.best_model == "TFT"
