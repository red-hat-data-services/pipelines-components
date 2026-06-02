"""Tests for component-local status tracking."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from kfp_components.components.training.automl.shared.component_status import (
    COMPONENT_STATUS_FILENAME,
    ComponentStatusTracker,
    load_component_status,
)


class TestComponentStatusTracker:
    """Tests for ComponentStatusTracker."""

    def test_record_and_save(self, tmp_path: Path) -> None:
        """save() writes stages and component_id to component_status.json."""
        tracker = ComponentStatusTracker(str(tmp_path), "automl_data_loader")
        tracker.record("validate_inputs", "started", rows=5)
        tracker.record("validate_inputs", "completed")
        tracker.save()

        data = json.loads((tmp_path / COMPONENT_STATUS_FILENAME).read_text(encoding="utf-8"))
        assert data["component_id"] == "automl_data_loader"
        assert len(data["stages"]) == 1
        assert data["stages"][0]["status"] == "completed"
        assert data["stages"][0]["rows"] == 5

    def test_mark_active_failed_marks_started_stage(self, tmp_path: Path) -> None:
        """mark_active_failed() updates the in-progress stage to failed."""
        tracker = ComponentStatusTracker(str(tmp_path), "autogluon_models_training")
        tracker.record("load_data", "completed")
        tracker.record("model_selection", "started")
        tracker.mark_active_failed("fit timeout")
        tracker.save()

        data = json.loads((tmp_path / COMPONENT_STATUS_FILENAME).read_text(encoding="utf-8"))
        assert data["stages"][-1]["status"] == "failed"
        assert data["stages"][-1]["error"] == "fit timeout"

    def test_save_best_effort_swallows_io_errors(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """save_best_effort() logs I/O errors instead of raising."""
        tracker = ComponentStatusTracker(str(tmp_path), "leaderboard_evaluation")
        tracker.record("build_leaderboard", "started")

        def _raise_save() -> None:
            raise OSError("disk full")

        monkeypatch.setattr(tracker, "save", _raise_save)
        tracker.save_best_effort()

    def test_context_manager_marks_failed_and_saves(self, tmp_path: Path) -> None:
        """Context manager marks active stage failed and persists status on exception."""
        with pytest.raises(RuntimeError, match="boom"):
            with ComponentStatusTracker(str(tmp_path), "automl_data_loader") as status:
                status.record("read_and_sample", "started")
                raise RuntimeError("boom")

        data = load_component_status(str(tmp_path))
        assert data["stages"][-1]["status"] == "failed"
        assert "boom" in data["stages"][-1]["error"]

    def test_context_manager_saves_on_success(self, tmp_path: Path) -> None:
        """Context manager persists status when the block completes normally."""
        with ComponentStatusTracker(str(tmp_path), "automl_data_loader") as status:
            status.record("write_outputs", "completed")

        assert (tmp_path / COMPONENT_STATUS_FILENAME).exists()

    def test_stage_context_manager_records_completed(self, tmp_path: Path) -> None:
        """stage() records started then completed when no exception is raised."""
        tracker = ComponentStatusTracker(str(tmp_path), "automl_data_loader")
        with tracker.stage("split"):
            pass
        tracker.save()

        data = json.loads((tmp_path / COMPONENT_STATUS_FILENAME).read_text(encoding="utf-8"))
        assert data["stages"][-1]["id"] == "split"
        assert data["stages"][-1]["status"] == "completed"

    def test_stage_context_manager_records_failed(self, tmp_path: Path) -> None:
        """stage() records failed when an exception escapes the block."""
        tracker = ComponentStatusTracker(str(tmp_path), "automl_data_loader")
        with pytest.raises(ValueError, match="bad split"):
            with tracker.stage("split"):
                raise ValueError("bad split")
        tracker.save()

        data = json.loads((tmp_path / COMPONENT_STATUS_FILENAME).read_text(encoding="utf-8"))
        assert data["stages"][-1]["status"] == "failed"
        assert "bad split" in data["stages"][-1]["error"]
