"""Tests for shared leaderboard utility functions in leaderboard_utils.py."""

from pathlib import Path
from unittest import mock

import pytest

# Importable via the sys.path insertion in components/training/automl/conftest.py,
# which replicates what KFP does at container runtime (adding the embedded
# artifact directory to sys.path).
from ..leaderboard_utils import _build_leaderboard_html, _build_leaderboard_table, _round_metrics


@pytest.fixture()
def template_path():
    """Path to the shared leaderboard HTML template."""
    return Path(__file__).resolve().parent.parent / "leaderboard_html_template.html"


class TestRoundMetrics:
    """Tests for _round_metrics."""

    def test_rounds_to_4_decimal_places(self):
        """Numeric values are rounded to 4 decimal places by default."""
        metrics = {"rmse": 0.123456789, "accuracy": 0.987654321}
        result = _round_metrics(metrics)
        assert result["rmse"] == 0.1235
        assert result["accuracy"] == 0.9877

    def test_preserves_non_numeric(self):
        """Non-numeric values (strings, None) pass through unchanged."""
        metrics = {"rmse": 0.123456789, "description": "some_text", "label": None}
        result = _round_metrics(metrics)
        assert result["rmse"] == 0.1235
        assert result["description"] == "some_text"
        assert result["label"] is None

    def test_handles_integers(self):
        """Integer values are preserved as integers after rounding."""
        metrics = {"count": 42, "score": 0.123456789}
        result = _round_metrics(metrics)
        assert result["count"] == 42
        assert result["score"] == 0.1235

    def test_empty_dict(self):
        """Empty metrics dict returns empty dict."""
        assert _round_metrics({}) == {}

    def test_custom_decimals(self):
        """Custom decimals argument controls rounding precision."""
        metrics = {"rmse": 0.123456789}
        assert _round_metrics(metrics, decimals=2)["rmse"] == 0.12
        assert _round_metrics(metrics, decimals=6)["rmse"] == 0.123457

    def test_negative_values(self):
        """Negative values (e.g. AutoGluon negated metrics) are rounded correctly."""
        metrics = {"MASE": -0.123456789, "WAPE": -0.987654321}
        result = _round_metrics(metrics)
        assert result["MASE"] == -0.1235
        assert result["WAPE"] == -0.9877


class TestBuildLeaderboardTable:
    """Tests for _build_leaderboard_table."""

    def _make_df(self, rows, columns, index_name="rank"):
        """Build a mock DataFrame usable by _build_leaderboard_table."""
        df = mock.MagicMock()
        df.columns = columns
        df.index.name = index_name
        df.iterrows.return_value = list(rows)
        return df

    def test_basic_html_structure(self):
        """Output is a valid table with thead and tbody."""
        columns = ["model", "rmse", "notebook", "predictor"]
        rows = [(1, {"model": "M1", "rmse": 0.5, "notebook": "http://nb", "predictor": "http://pred"})]
        html = _build_leaderboard_table(self._make_df(rows, columns))
        assert html.startswith("<table>")
        assert html.endswith("</table>")
        assert "<thead>" in html
        assert "<tbody>" in html

    def test_header_contains_rank_and_metric_columns(self):
        """Header row has rank, metric columns, Notebook, Predictor."""
        columns = ["model", "rmse", "notebook", "predictor"]
        rows = [(1, {"model": "M1", "rmse": 0.5, "notebook": "nb", "predictor": "pred"})]
        html = _build_leaderboard_table(self._make_df(rows, columns))
        assert "<th>rank</th>" in html
        assert "<th>model</th>" in html
        assert "<th>rmse</th>" in html
        assert "<th>Notebook</th>" in html
        assert "<th>Predictor</th>" in html

    def test_notebook_and_predictor_not_regular_columns(self):
        """'notebook' and 'predictor' do not appear as plain <th> column headers."""
        columns = ["model", "notebook", "predictor"]
        rows = [(1, {"model": "M1", "notebook": "nb", "predictor": "pred"})]
        html = _build_leaderboard_table(self._make_df(rows, columns))
        assert "<th>notebook</th>" not in html
        assert "<th>predictor</th>" not in html

    def test_uri_cells_have_popover_structure(self):
        """Notebook and predictor cells use uri-cell/uri-link/uri-popover markup."""
        columns = ["model", "notebook", "predictor"]
        rows = [(1, {"model": "M1", "notebook": "http://example.com/nb", "predictor": "http://example.com/pred"})]
        html = _build_leaderboard_table(self._make_df(rows, columns))
        assert 'class="uri-cell"' in html
        assert 'class="uri-link"' in html
        assert 'class="uri-popover"' in html
        assert 'data-uri="http://example.com/nb"' in html
        assert 'data-uri="http://example.com/pred"' in html

    def test_html_special_characters_escaped(self):
        """Values containing HTML special characters are escaped."""
        columns = ["model", "notebook", "predictor"]
        rows = [(1, {"model": 'Model<1>&"', "notebook": "http://nb", "predictor": "http://pred"})]
        html = _build_leaderboard_table(self._make_df(rows, columns))
        assert "Model&lt;1&gt;&amp;&quot;" in html
        assert "Model<1>&" not in html

    def test_multiple_rows_all_present(self):
        """All rows appear in the output and row count matches."""
        columns = ["model", "rmse", "notebook", "predictor"]
        rows = [
            (1, {"model": "ModelA", "rmse": 0.3, "notebook": "nb1", "predictor": "pred1"}),
            (2, {"model": "ModelB", "rmse": 0.5, "notebook": "nb2", "predictor": "pred2"}),
            (3, {"model": "ModelC", "rmse": 0.8, "notebook": "nb3", "predictor": "pred3"}),
        ]
        html = _build_leaderboard_table(self._make_df(rows, columns))
        assert "ModelA" in html
        assert "ModelB" in html
        assert "ModelC" in html
        # 3 data rows + 1 header row in <thead>
        assert html.count("<tr>") == 4

    def test_row_index_included_as_first_cell(self):
        """The row index (rank) is rendered as the first <td> in each row."""
        columns = ["model", "notebook", "predictor"]
        rows = [(7, {"model": "M1", "notebook": "nb", "predictor": "pred"})]
        html = _build_leaderboard_table(self._make_df(rows, columns))
        assert "<td>7</td>" in html

    def test_custom_index_name(self):
        """A custom index name is used instead of 'rank'."""
        columns = ["model", "notebook", "predictor"]
        rows = [(1, {"model": "M1", "notebook": "nb", "predictor": "pred"})]
        html = _build_leaderboard_table(self._make_df(rows, columns, index_name="position"))
        assert "<th>position</th>" in html
        assert "<th>rank</th>" not in html


class TestBuildLeaderboardHtml:
    """Tests for _build_leaderboard_html."""

    def test_all_placeholders_replaced(self, template_path):
        """No placeholder tokens remain after substitution."""
        html = _build_leaderboard_html(
            template_path=template_path,
            table_html="<table></table>",
            eval_metric="rmse",
            best_model_name="BestModel",
            num_models=3,
        )
        assert "__TABLE_HTML__" not in html
        assert "__NUM_MODELS__" not in html
        assert "__EVAL_METRIC__" not in html
        assert "__BEST_MODEL_NAME__" not in html

    def test_values_appear_in_output(self, template_path):
        """Substituted values are present in the rendered HTML."""
        html = _build_leaderboard_html(
            template_path=template_path,
            table_html="<table>MY_TABLE</table>",
            eval_metric="accuracy",
            best_model_name="TopModel",
            num_models=5,
        )
        assert "MY_TABLE" in html
        assert "accuracy" in html
        assert "TopModel" in html
        assert "5" in html

    def test_output_is_complete_html_document(self, template_path):
        """Output contains standard HTML document markers."""
        html = _build_leaderboard_html(
            template_path=template_path,
            table_html="<table></table>",
            eval_metric="r2",
            best_model_name="M1",
            num_models=1,
        )
        assert "<!DOCTYPE html>" in html
        assert "<html" in html
        assert "</html>" in html

    def test_missing_template_raises(self, tmp_path):
        """FileNotFoundError is raised when the template path does not exist."""
        with pytest.raises(FileNotFoundError):
            _build_leaderboard_html(
                template_path=tmp_path / "nonexistent.html",
                table_html="<table></table>",
                eval_metric="rmse",
                best_model_name="M",
                num_models=1,
            )
