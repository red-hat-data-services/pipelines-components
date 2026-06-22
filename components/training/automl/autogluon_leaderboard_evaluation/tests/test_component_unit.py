"""Tests for the leaderboard_evaluation component."""

import json
from pathlib import Path
from unittest import mock

import pytest

from ..component import leaderboard_evaluation


def _make_component_status_artifact(tmp_path):
    art = mock.MagicMock()
    art.path = str(tmp_path / "component_status_out")
    art.metadata = {}
    return art


_DEFAULT_COMPONENT_STATUS = _make_component_status_artifact(Path("/tmp"))


def _make_models_artifact(
    base_path: str | Path,
    model_names: list[str],
    *,
    uri: str = "http://example.com/artifacts",
    metadata: dict | None = None,
):
    """Build a mock combined models artifact (path + model_names in metadata)."""
    m = mock.MagicMock()
    m.path = str(base_path)
    m.uri = uri
    meta = dict(metadata) if metadata is not None else {}
    if "model_names" not in meta:
        meta["model_names"] = json.dumps(model_names)
    m.metadata = meta
    return m


@pytest.fixture()
def create_model_dir(tmp_path):
    """Factory fixture to create model artifact directories with metrics.json."""
    _counter = [0]

    def _create(metrics_dict, model_name="Model1"):
        model_dir = tmp_path / f"model_artifact_{_counter[0]}"
        _counter[0] += 1
        model_dir.mkdir()
        metrics_dir = model_dir / model_name / "metrics"
        metrics_dir.mkdir(parents=True)
        (metrics_dir / "metrics.json").write_text(json.dumps(metrics_dict))
        return str(model_dir)

    return _create


@pytest.fixture()
def html_output_path(tmp_path):
    """Provide a temporary HTML output path."""
    return str(tmp_path / "leaderboard.html")


def _make_mock_sorted_df(rows, columns):
    """Build a mock sorted DataFrame with the given rows and columns."""
    mock_df_sorted = mock.MagicMock()
    mock_df_sorted.__len__ = lambda self: len(rows)
    mock_df_sorted.index.name = "rank"
    mock_df_sorted.columns = columns
    mock_df_sorted.iterrows.return_value = list(rows)
    # Support .iloc[0]["model"] for best_model extraction
    row_dicts = [r[1] for r in rows]
    mock_df_sorted.iloc.__getitem__ = lambda _, idx: row_dicts[idx]
    return mock_df_sorted


class TestLeaderboardEvaluationUnitTests:
    """Unit tests for component logic."""

    @mock.patch("pandas.DataFrame")
    def test_single_model(self, mock_dataframe_class, create_model_dir, html_output_path):
        """Test leaderboard with a single model: return value, metadata, HTML output."""
        metrics = {"root_mean_squared_error": 0.5, "mean_absolute_error": 0.4, "r2": 0.9}
        model_dir = create_model_dir(metrics, model_name="Model1")

        columns = ["model", "root_mean_squared_error", "mean_absolute_error", "r2", "notebook", "predictor"]
        rows = [
            (
                1,
                {
                    "model": "Model1",
                    "root_mean_squared_error": 0.5,
                    "mean_absolute_error": 0.4,
                    "r2": 0.9,
                    "notebook": "http://example.com/artifacts/Model1/notebooks/automl_predictor_notebook.ipynb",
                    "predictor": "http://example.com/artifacts/Model1/predictor",
                },
            ),
        ]
        mock_df_sorted = _make_mock_sorted_df(rows, columns)
        mock_df = mock.MagicMock()
        mock_df.sort_values.return_value = mock_df_sorted
        mock_dataframe_class.return_value = mock_df

        models_artifact = _make_models_artifact(model_dir, ["Model1"])

        mock_html = mock.MagicMock()
        mock_html.path = html_output_path
        mock_html.metadata = {}

        result = leaderboard_evaluation.python_func(
            models_artifact=models_artifact,
            eval_metric="root_mean_squared_error",
            html_artifact=mock_html,
            component_status=_DEFAULT_COMPONENT_STATUS,
        )

        # Verify DataFrame was constructed with correct data
        mock_dataframe_class.assert_called_once()
        call_args = mock_dataframe_class.call_args[0][0]
        assert len(call_args) == 1
        assert call_args[0]["model"] == "Model1"
        assert call_args[0]["root_mean_squared_error"] == 0.5
        assert call_args[0]["mean_absolute_error"] == 0.4
        assert call_args[0]["r2"] == 0.9
        assert (
            call_args[0]["notebook"] == "http://example.com/artifacts/Model1/notebooks/automl_predictor_notebook.ipynb"
        )
        assert call_args[0]["predictor"] == "http://example.com/artifacts/Model1/predictor"

        # Verify sort
        mock_df.sort_values.assert_called_once_with(by="root_mean_squared_error", ascending=False)

        # Verify best_model return value
        assert result.best_model == "Model1"

        # Verify HTML artifact metadata
        assert mock_html.metadata["display_name"] == "automl_leaderboard"
        assert "data" in mock_html.metadata

        # Verify HTML file content
        html = Path(html_output_path).read_text()
        assert "Notebook" in html
        assert "Predictor" in html
        assert "automl_predictor_notebook.ipynb" in html
        assert "Model1" in html
        assert "uri-cell" in html
        assert "uri-link" in html

    @mock.patch("pandas.DataFrame")
    def test_multiple_models(self, mock_dataframe_class, tmp_path, html_output_path):
        """Test leaderboard with multiple models and best_model selection."""
        metrics_list = [
            {"root_mean_squared_error": 0.8, "mean_absolute_error": 0.6},
            {"root_mean_squared_error": 0.3, "mean_absolute_error": 0.2},
            {"root_mean_squared_error": 0.5, "mean_absolute_error": 0.4},
        ]
        combined_root = tmp_path / "combined_models"
        for i, m in enumerate(metrics_list):
            name = f"Model{i + 1}"
            metrics_dir = combined_root / name / "metrics"
            metrics_dir.mkdir(parents=True)
            (metrics_dir / "metrics.json").write_text(json.dumps(m))

        columns = ["model", "root_mean_squared_error", "mean_absolute_error", "notebook", "predictor"]
        rows = [
            (
                1,
                {
                    "model": "Model2",
                    "root_mean_squared_error": 0.3,
                    "mean_absolute_error": 0.2,
                    "notebook": "nb2",
                    "predictor": "p2",
                },
            ),
            (
                2,
                {
                    "model": "Model3",
                    "root_mean_squared_error": 0.5,
                    "mean_absolute_error": 0.4,
                    "notebook": "nb3",
                    "predictor": "p3",
                },
            ),
            (
                3,
                {
                    "model": "Model1",
                    "root_mean_squared_error": 0.8,
                    "mean_absolute_error": 0.6,
                    "notebook": "nb1",
                    "predictor": "p1",
                },
            ),
        ]
        mock_df_sorted = _make_mock_sorted_df(rows, columns)
        mock_df = mock.MagicMock()
        mock_df.sort_values.return_value = mock_df_sorted
        mock_dataframe_class.return_value = mock_df

        models_artifact = _make_models_artifact(combined_root, ["Model1", "Model2", "Model3"])

        mock_html = mock.MagicMock()
        mock_html.path = html_output_path
        mock_html.metadata = {}

        result = leaderboard_evaluation.python_func(
            models_artifact=models_artifact,
            eval_metric="root_mean_squared_error",
            html_artifact=mock_html,
            component_status=_make_component_status_artifact(tmp_path),
        )

        # Verify all models were passed to DataFrame
        call_args = mock_dataframe_class.call_args[0][0]
        assert len(call_args) == 3
        assert call_args[0]["model"] == "Model1"
        assert call_args[1]["model"] == "Model2"
        assert call_args[2]["model"] == "Model3"

        # Best model is first after sorting
        assert result.best_model == "Model2"

        mock_df.sort_values.assert_called_once_with(by="root_mean_squared_error", ascending=False)

        # Verify HTML was written with all model data
        html = Path(html_output_path).read_text()
        assert "Notebook" in html and "Predictor" in html
        assert "Model2" in html
        assert "Model3" in html
        assert "Model1" in html

    def test_empty_model_names_raises(self, html_output_path):
        """Empty or missing ``model_names`` in artifact metadata raises ``KeyError``."""
        models_artifact = _make_models_artifact("/tmp/some_path", [], metadata={"model_names": "[]"})

        mock_html = mock.MagicMock()
        mock_html.path = html_output_path
        mock_html.metadata = {}

        with pytest.raises(KeyError, match="model_names"):
            leaderboard_evaluation.python_func(
                models_artifact=models_artifact,
                eval_metric="rmse",
                html_artifact=mock_html,
                component_status=_DEFAULT_COMPONENT_STATUS,
            )

        models_artifact_default = _make_models_artifact("/tmp/some_path", [], metadata={})

        with pytest.raises(KeyError, match="model_names"):
            leaderboard_evaluation.python_func(
                models_artifact=models_artifact_default,
                eval_metric="rmse",
                html_artifact=mock_html,
                component_status=_DEFAULT_COMPONENT_STATUS,
            )

    def test_leaderboard_evaluation_rejects_empty_eval_metric(self):
        """Test that TypeError is raised when eval_metric is empty or not a string."""
        models_artifact = _make_models_artifact("/tmp/model", ["Model1"])
        mock_html = mock.MagicMock()
        mock_html.path = "/tmp/out.html"

        with pytest.raises(TypeError, match=r"eval_metric must be a non-empty string\."):
            leaderboard_evaluation.python_func(
                models_artifact=models_artifact,
                eval_metric="",
                html_artifact=mock_html,
                component_status=_DEFAULT_COMPONENT_STATUS,
            )

        with pytest.raises(TypeError, match=r"eval_metric must be a non-empty string\."):
            leaderboard_evaluation.python_func(
                models_artifact=models_artifact,
                eval_metric="   ",
                html_artifact=mock_html,
                component_status=_DEFAULT_COMPONENT_STATUS,
            )

    def test_missing_metrics_file_raises(self, tmp_path, html_output_path):
        """Missing metrics.json is skipped; if no models remain, raises ValueError."""
        model_dir = tmp_path / "model_artifact_empty"
        model_dir.mkdir()

        models_artifact = _make_models_artifact(model_dir, ["Model1"])

        mock_html = mock.MagicMock()
        mock_html.path = html_output_path
        mock_html.metadata = {}

        with pytest.raises(ValueError, match="No valid model artifacts found"):
            leaderboard_evaluation.python_func(
                models_artifact=models_artifact,
                eval_metric="rmse",
                html_artifact=mock_html,
                component_status=_make_component_status_artifact(tmp_path),
            )

    def test_component_imports_correctly(self):
        """Test that the component can be imported and has required attributes."""
        assert callable(leaderboard_evaluation)
        assert hasattr(leaderboard_evaluation, "python_func")
        assert hasattr(leaderboard_evaluation, "component_spec")


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
