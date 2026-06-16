"""Tests for back_testing.json matplotlib charts."""

from __future__ import annotations

import ast

import pytest

from ..back_testing_charts import (
    _draw_forecast,
    _interval_label,
    _show_per_window_metrics,
    forecast_data_to_frame,
    mean_window_metric,
    notebook_backtest_charts_source,
    per_window_metrics_table,
    pick_window_metric,
    render_back_testing_charts,
)


class TestBackTestingCharts:
    """Tests for backtest chart helpers."""

    @staticmethod
    def _sample_forecast_rows():
        return [
            {
                "timestamp": "2025-01-03T00:00:00Z",
                "actual": 100.0,
                "predicted": 105.0,
                "lower_bound": 95.0,
                "upper_bound": 115.0,
                "lower_quantile": 0.1,
                "upper_quantile": 0.9,
            },
            {
                "timestamp": "2025-01-04T00:00:00Z",
                "actual": 110.0,
                "predicted": 108.0,
                "lower_bound": 98.0,
                "upper_bound": 118.0,
                "lower_quantile": 0.1,
                "upper_quantile": 0.9,
            },
        ]

    def test_per_window_metrics_table_includes_cutoff(self):
        """Per-window table exposes AutoGluon-style cutoff identifiers."""
        per_window = [
            {
                "window_id": 0,
                "cutoff": -6,
                "test_start": "2025-03-01",
                "test_end": "2025-03-02",
                "metrics": {"MASE": 0.38},
            },
            {
                "window_id": 1,
                "cutoff": -4,
                "test_start": "2025-03-03",
                "test_end": "2025-03-04",
                "metrics": {"MASE": 0.45},
            },
        ]
        table = per_window_metrics_table(per_window, "MASE")
        assert list(table["cutoff"]) == [-6, -4]
        assert list(table["MASE"]) == [0.38, 0.45]

    def test_mean_window_metric_averages_finite_values(self):
        """Overall summary uses the mean of per-window eval metric values."""
        per_window = [
            {"window_id": 0, "metrics": {"MASE": 0.38}},
            {"window_id": 1, "metrics": {"MASE": 0.42}},
        ]
        assert mean_window_metric(per_window, "MASE") == pytest.approx(0.4)

    def test_show_per_window_metrics_prints_cutoff_lines(self, capsys):
        """Output matches AutoGluon tutorial print format."""
        per_window = [
            {
                "window_id": 0,
                "cutoff": -6,
                "test_start": "2025-03-01",
                "test_end": "2025-03-02",
                "metrics": {"MASE": 0.38},
            },
        ]
        _show_per_window_metrics(per_window, "MASE")
        output = capsys.readouterr().out
        assert "Cutoff -6: MASE = 0.38" in output

    def test_show_per_window_metrics_prints_overall_summary(self, capsys):
        """Overall line aggregates window scores across all series."""
        per_window = [
            {"window_id": 0, "cutoff": -6, "metrics": {"MASE": 0.38}},
            {"window_id": 1, "cutoff": -4, "metrics": {"MASE": 0.42}},
        ]
        _show_per_window_metrics(per_window, "MASE", num_series_evaluated=12)
        output = capsys.readouterr().out
        assert "Overall MASE (mean across 2 validation windows, all series): 0.4" in output
        assert "Series evaluated: 12" in output

    def test_forecast_data_to_frame_parses_timestamps(self):
        """Forecast rows are sorted by timestamp."""
        frame = forecast_data_to_frame(self._sample_forecast_rows())
        assert "lower_quantile" in frame.columns
        assert frame["timestamp"].is_monotonic_increasing

    def test_interval_label_uses_quantile_metadata(self):
        """Interval legend matches AutoGluon-style P10-P90 labeling."""
        frame = forecast_data_to_frame(self._sample_forecast_rows())
        assert _interval_label(frame) == "10%-90% interval"

    def test_draw_forecast_adds_cutoff_line(self):
        """Cutoff vline is drawn at the validation window start."""
        pytest.importorskip("matplotlib")
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        figure, axis = plt.subplots()
        _draw_forecast(
            axis,
            self._sample_forecast_rows(),
            title="Window 0",
            target="sales",
            cutoff_start="2025-01-03",
        )
        assert any(line.get_linestyle() == "--" for line in axis.lines)
        plt.close(figure)

    def test_pick_window_metric_prefers_eval_metric(self):
        """Metric lookup matches normalized eval metric names."""
        metrics = {"MASE": 0.42, "MAPE": 5.0}
        assert pick_window_metric(metrics, "MASE") == 0.42
        assert pick_window_metric(metrics, "-MASE") == 0.42

    def test_notebook_backtest_charts_source_compiles(self):
        """Notebook injection source is valid Python."""
        ast.parse(notebook_backtest_charts_source())

    def test_render_back_testing_charts_runs(self):
        """Orchestrator renders charts for a minimal payload."""
        pytest.importorskip("matplotlib")
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        payload = {
            "eval_metric": "MASE",
            "target": "sales",
            "per_window_metrics": [
                {
                    "window_id": 0,
                    "cutoff": -2,
                    "test_start": "2025-01-01",
                    "test_end": "2025-01-02",
                    "metrics": {"MASE": 0.4},
                },
                {
                    "window_id": 1,
                    "cutoff": -4,
                    "test_start": "2025-01-03",
                    "test_end": "2025-01-04",
                    "metrics": {"MASE": 0.5},
                },
            ],
            "series_analysis": {
                "num_series_evaluated": 5,
                "best_performer": {
                    "item_id": "A",
                    "windows": [
                        {
                            "window_id": 0,
                            "metrics": {"MAPE": 5.0},
                            "forecast_data": self._sample_forecast_rows(),
                        },
                        {
                            "window_id": 1,
                            "metrics": {"MAPE": 6.0},
                            "forecast_data": self._sample_forecast_rows(),
                        },
                    ],
                },
                "worst_performer": None,
            },
        }
        render_back_testing_charts(payload)
        plt.close("all")
