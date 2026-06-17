"""Tests for component-local status tracking."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from kfp_components.components.training.automl.shared.component_status import (
    COMPONENT_STATUS_FILENAME,
    ComponentStatusEncoder,
    ComponentStatusTracker,
    load_component_status,
    utc_now_z,
)


class TestComponentStatusTracker:
    """Tests for ComponentStatusTracker."""

    def test_record_and_save(self, tmp_path: Path) -> None:
        """save() writes stages and component_id to component_status.json."""
        tracker = ComponentStatusTracker(str(tmp_path), "automl_data_loader")
        tracker.record("prepare_data", "started", rows=5)
        tracker.record("prepare_data", "completed")
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
                status.record("prepare_data", "started")
                raise RuntimeError("boom")

        data = load_component_status(str(tmp_path))
        assert data["stages"][-1]["status"] == "failed"
        assert "boom" in data["stages"][-1]["error"]

    def test_context_manager_saves_on_success(self, tmp_path: Path) -> None:
        """Context manager persists status when the block completes normally."""
        with ComponentStatusTracker(str(tmp_path), "automl_data_loader") as status:
            status.record("split_and_export", "completed")

        assert (tmp_path / COMPONENT_STATUS_FILENAME).exists()

    def test_stage_context_manager_records_completed(self, tmp_path: Path) -> None:
        """stage() records started then completed when no exception is raised."""
        tracker = ComponentStatusTracker(str(tmp_path), "automl_data_loader")
        with tracker.stage("split_and_export"):
            pass
        tracker.save()

        data = json.loads((tmp_path / COMPONENT_STATUS_FILENAME).read_text(encoding="utf-8"))
        assert data["stages"][-1]["id"] == "split_and_export"
        assert data["stages"][-1]["status"] == "completed"

    def test_stage_context_manager_records_failed(self, tmp_path: Path) -> None:
        """stage() records failed when an exception escapes the block."""
        tracker = ComponentStatusTracker(str(tmp_path), "automl_data_loader")
        with pytest.raises(ValueError, match="bad split"):
            with tracker.stage("split_and_export"):
                raise ValueError("bad split")
        tracker.save()

        data = json.loads((tmp_path / COMPONENT_STATUS_FILENAME).read_text(encoding="utf-8"))
        assert data["stages"][-1]["status"] == "failed"
        assert "bad split" in data["stages"][-1]["error"]

    def test_disabled_tracker_skips_save(self, tmp_path: Path) -> None:
        """When artifact_path is None, save() is a no-op."""
        tracker = ComponentStatusTracker(None, "automl_data_loader")
        tracker.record("prepare_data", "completed")
        tracker.save()
        assert not (tmp_path / COMPONENT_STATUS_FILENAME).exists()

    def test_stage_skips_auto_complete_when_completed_inside_block(self, tmp_path: Path) -> None:
        """stage() does not overwrite a completed record written inside the block."""
        tracker = ComponentStatusTracker(str(tmp_path), "autogluon_models_training")
        with tracker.stage("model_selection", steps=["feature_engineering"]):
            tracker.record("model_selection", "completed", top_n=3)
        tracker.save()

        data = load_component_status(str(tmp_path))
        model_stage = next(stage for stage in data["stages"] if stage["id"] == "model_selection")
        assert model_stage["status"] == "completed"
        assert model_stage["top_n"] == 3

    def test_utc_now_z_ends_with_z(self) -> None:
        """Timestamps use UTC ISO-8601 with Z suffix."""
        assert utc_now_z().endswith("Z")


class TestComponentStatusEncoder:
    """Tests for JSON encoding of status metadata values."""

    def test_encodes_datetime_path_bytes_and_set(self) -> None:
        """Known non-JSON types are converted for serialization."""
        from datetime import UTC, datetime

        encoder = ComponentStatusEncoder()
        assert encoder.default(Path("/tmp/out")) == "/tmp/out"
        assert encoder.default(b"abc") == "YWJj"
        assert encoder.default({1, 2}) == [1, 2]
        encoded = encoder.default(datetime(2026, 6, 10, 12, 0, 0, tzinfo=UTC))
        assert encoded.endswith("Z")

    def test_unknown_type_raises_type_error(self) -> None:
        """Unsupported metadata types fail fast instead of being stringified."""
        encoder = ComponentStatusEncoder()
        with pytest.raises(TypeError):
            encoder.default(object())


class TestLoadComponentStatus:
    """Tests for load_component_status edge cases."""

    def test_corrupt_json_returns_empty_dict(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        """Corrupt status files return {} and log a warning."""
        status_file = tmp_path / COMPONENT_STATUS_FILENAME
        status_file.write_text("{not json", encoding="utf-8")
        with caplog.at_level("WARNING"):
            assert load_component_status(str(tmp_path)) == {}
        assert "Failed to load status" in caplog.text
