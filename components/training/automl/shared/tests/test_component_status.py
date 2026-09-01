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

_CANONICAL_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "required": ["component_id", "started_at", "stages", "metadata"],
    "additionalProperties": False,
    "properties": {
        "component_id": {"type": "string", "pattern": "^[a-z][a-z0-9_]*$"},
        "started_at": {"type": "string", "format": "date-time"},
        "completed_at": {"type": "string", "format": "date-time"},
        "metadata": {
            "type": "object",
            "required": ["display_name"],
            "additionalProperties": True,
            "properties": {
                "display_name": {"type": "string", "minLength": 1},
            },
        },
        "stages": {
            "type": "array",
            "items": {"$ref": "#/$defs/stage"},
        },
    },
    "$defs": {
        "statusMessage": {
            "type": "object",
            "required": ["level", "text"],
            "additionalProperties": False,
            "properties": {
                "level": {"type": "string", "enum": ["info", "warning", "error"]},
                "text": {"type": "string", "minLength": 1},
            },
        },
        "stageStatus": {
            "type": "object",
            "required": ["state"],
            "additionalProperties": False,
            "properties": {
                "state": {"type": "string", "enum": ["started", "running", "completed", "failed"]},
                "step": {"type": "string"},
                "message": {"$ref": "#/$defs/statusMessage"},
                "running_at": {"type": "string", "format": "date-time"},
            },
        },
        "stage": {
            "type": "object",
            "required": ["id", "status"],
            "additionalProperties": False,
            "properties": {
                "id": {"type": "string", "pattern": "^[a-z][a-z0-9_]*$"},
                "status": {"$ref": "#/$defs/stageStatus"},
                "metrics": {
                    "type": "object",
                    "additionalProperties": {
                        "type": ["string", "number", "boolean", "array", "null"],
                        "items": {"type": ["string", "number", "boolean"]},
                    },
                },
                "error": {"type": "string"},
            },
        },
    },
}


@pytest.fixture()
def canonical_schema() -> dict:
    """Return the canonical component_status JSON Schema."""
    return _CANONICAL_SCHEMA


def _save_and_load(tracker: ComponentStatusTracker) -> dict:
    """Save tracker to disk and return the parsed JSON."""
    tracker.save()
    return json.loads((Path(tracker.artifact_path) / COMPONENT_STATUS_FILENAME).read_text(encoding="utf-8"))


def _assert_schema_valid(instance: dict, schema: dict) -> None:
    """Validate instance against schema (lazy-import jsonschema for import guard)."""
    from jsonschema import validate

    validate(instance=instance, schema=schema)


class TestComponentStatusTracker:
    """Tests for ComponentStatusTracker."""

    def test_record_and_save(self, tmp_path: Path) -> None:
        """save() writes stages with nested status and metrics to component_status.json."""
        tracker = ComponentStatusTracker(str(tmp_path), "automl_data_loader")
        tracker.record("prepare_data", "started", metrics={"rows": 5})
        tracker.record("prepare_data", "completed")
        tracker.save()

        data = json.loads((tmp_path / COMPONENT_STATUS_FILENAME).read_text(encoding="utf-8"))
        assert data["component_id"] == "automl_data_loader"
        assert len(data["stages"]) == 1
        assert data["stages"][0]["status"]["state"] == "completed"
        assert data["stages"][0]["metrics"]["rows"] == 5

    def test_mark_active_failed_marks_started_stage(self, tmp_path: Path) -> None:
        """mark_active_failed() updates the in-progress stage to failed."""
        tracker = ComponentStatusTracker(str(tmp_path), "autogluon_models_training")
        tracker.record("model_selection", "started")
        tracker.mark_active_failed("fit timeout")
        tracker.save()

        data = json.loads((tmp_path / COMPONENT_STATUS_FILENAME).read_text(encoding="utf-8"))
        assert data["stages"][-1]["status"]["state"] == "failed"
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
        assert data["stages"][-1]["status"]["state"] == "failed"
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
        assert data["stages"][-1]["status"]["state"] == "completed"

    def test_stage_context_manager_records_failed(self, tmp_path: Path) -> None:
        """stage() records failed when an exception escapes the block."""
        tracker = ComponentStatusTracker(str(tmp_path), "automl_data_loader")
        with pytest.raises(ValueError, match="bad split"):
            with tracker.stage("split_and_export"):
                raise ValueError("bad split")
        tracker.save()

        data = json.loads((tmp_path / COMPONENT_STATUS_FILENAME).read_text(encoding="utf-8"))
        assert data["stages"][-1]["status"]["state"] == "failed"
        assert "bad split" in data["stages"][-1]["error"]

    def test_stage_skips_auto_complete_when_completed_inside_block(self, tmp_path: Path) -> None:
        """stage() does not overwrite a completed record written inside the block."""
        tracker = ComponentStatusTracker(str(tmp_path), "autogluon_models_training")
        with tracker.stage("model_selection"):
            tracker.record("model_selection", "completed", metrics={"top_n": 3})
        tracker.save()

        data = load_component_status(str(tmp_path))
        model_stage = next(stage for stage in data["stages"] if stage["id"] == "model_selection")
        assert model_stage["status"]["state"] == "completed"
        assert model_stage["metrics"]["top_n"] == 3

    def test_utc_now_z_ends_with_z(self) -> None:
        """Timestamps use UTC ISO-8601 with Z suffix."""
        assert utc_now_z().endswith("Z")

    def test_running_state_sets_running_at(self, tmp_path: Path) -> None:
        """Recording state=running auto-sets running_at timestamp."""
        tracker = ComponentStatusTracker(str(tmp_path), "automl_data_loader")
        tracker.record("prepare_data", "running")
        tracker.save()

        data = json.loads((tmp_path / COMPONENT_STATUS_FILENAME).read_text(encoding="utf-8"))
        assert data["stages"][0]["status"]["state"] == "running"
        assert "running_at" in data["stages"][0]["status"]
        assert data["stages"][0]["status"]["running_at"].endswith("Z")

    def test_step_field_on_status(self, tmp_path: Path) -> None:
        """Step is placed inside the nested status object."""
        tracker = ComponentStatusTracker(str(tmp_path), "autogluon_models_training")
        tracker.record("model_selection", "running", step="model_training")
        tracker.save()

        data = json.loads((tmp_path / COMPONENT_STATUS_FILENAME).read_text(encoding="utf-8"))
        assert data["stages"][0]["status"]["step"] == "model_training"

    def test_message_field_on_status(self, tmp_path: Path) -> None:
        """Message is placed inside the nested status object."""
        tracker = ComponentStatusTracker(str(tmp_path), "automl_data_loader")
        tracker.record("prepare_data", "running", message={"level": "info", "text": "Loading data..."})
        tracker.save()

        data = json.loads((tmp_path / COMPONENT_STATUS_FILENAME).read_text(encoding="utf-8"))
        assert data["stages"][0]["status"]["message"] == {"level": "info", "text": "Loading data..."}

    def test_metrics_accumulate_across_records(self, tmp_path: Path) -> None:
        """Metrics from multiple record() calls on the same stage merge together."""
        tracker = ComponentStatusTracker(str(tmp_path), "automl_data_loader")
        tracker.record("prepare_data", "started", metrics={"rows": 100})
        tracker.record("prepare_data", "completed", metrics={"duplicates_dropped": 5})
        tracker.save()

        data = json.loads((tmp_path / COMPONENT_STATUS_FILENAME).read_text(encoding="utf-8"))
        assert data["stages"][0]["metrics"]["rows"] == 100
        assert data["stages"][0]["metrics"]["duplicates_dropped"] == 5

    def test_completed_at_set_when_finished(self, tmp_path: Path) -> None:
        """completed_at is set only when all stages are completed."""
        tracker = ComponentStatusTracker(str(tmp_path), "automl_data_loader")
        tracker.record("prepare_data", "completed")
        tracker.record("split_and_export", "completed")
        tracker.save()

        data = json.loads((tmp_path / COMPONENT_STATUS_FILENAME).read_text(encoding="utf-8"))
        assert "completed_at" in data

    def test_completed_at_not_set_while_running(self, tmp_path: Path) -> None:
        """completed_at is not set while stages are still in progress."""
        tracker = ComponentStatusTracker(str(tmp_path), "automl_data_loader")
        tracker.record("prepare_data", "completed")
        tracker.record("split_and_export", "started")
        tracker.save()

        data = json.loads((tmp_path / COMPONENT_STATUS_FILENAME).read_text(encoding="utf-8"))
        assert "completed_at" not in data

    def test_completed_at_preserved_across_repeated_saves(self, tmp_path: Path) -> None:
        """First terminal completed_at is kept when save() is called again."""
        tracker = ComponentStatusTracker(str(tmp_path), "automl_data_loader")
        tracker.record("prepare_data", "completed")
        tracker.save()
        first = json.loads((tmp_path / COMPONENT_STATUS_FILENAME).read_text(encoding="utf-8"))["completed_at"]

        tracker.save()
        second = json.loads((tmp_path / COMPONENT_STATUS_FILENAME).read_text(encoding="utf-8"))["completed_at"]
        assert second == first

    def test_failed_to_completed_clears_stale_error(self, tmp_path: Path) -> None:
        """Updating a failed stage to completed removes the previous error."""
        tracker = ComponentStatusTracker(str(tmp_path), "automl_data_loader")
        tracker.record("prepare_data", "failed", error="ValueError: boom")
        tracker.record("prepare_data", "completed")
        tracker.save()

        data = json.loads((tmp_path / COMPONENT_STATUS_FILENAME).read_text(encoding="utf-8"))
        assert data["stages"][0]["status"]["state"] == "completed"
        assert "error" not in data["stages"][0]

    def test_invalid_state_raises(self, tmp_path: Path) -> None:
        """record() rejects invalid state values."""
        tracker = ComponentStatusTracker(str(tmp_path), "automl_data_loader")
        with pytest.raises(ValueError, match="state must be one of"):
            tracker.record("prepare_data", "unknown_state")

    def test_no_flat_keys_on_stage(self, tmp_path: Path) -> None:
        """Stage objects only contain id, status, metrics, and error — no flat keys."""
        tracker = ComponentStatusTracker(str(tmp_path), "automl_data_loader")
        tracker.record("prepare_data", "completed", metrics={"rows": 10})
        tracker.save()

        data = json.loads((tmp_path / COMPONENT_STATUS_FILENAME).read_text(encoding="utf-8"))
        allowed_keys = {"id", "status", "metrics", "error"}
        actual_keys = set(data["stages"][0].keys())
        assert actual_keys <= allowed_keys, f"Unexpected stage keys: {actual_keys - allowed_keys}"


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


class TestSchemaCompliance:
    """Validate tracker output against the canonical JSON Schema."""

    def test_completed_data_loader(self, tmp_path: Path, canonical_schema: dict) -> None:
        """Completed data loader output validates against the schema."""
        tracker = ComponentStatusTracker(str(tmp_path), "automl_data_loader")
        tracker.set_metadata(display_name="Data Loader Status")
        tracker.record("prepare_data", "started")
        tracker.record(
            "prepare_data",
            "running",
            metrics={"sampling_method": "random", "source": "s3://bucket/file.csv"},
        )
        tracker.record(
            "prepare_data",
            "completed",
            metrics={"rows": 500, "duplicates_dropped": 3, "labels_dropped": 1},
        )
        tracker.record("split_and_export", "started")
        tracker.record(
            "split_and_export",
            "completed",
            metrics={"test_size": 0.2, "selection_train_size": 0.3, "stratify": False},
        )

        _assert_schema_valid(_save_and_load(tracker), canonical_schema)

    def test_completed_models_training(self, tmp_path: Path, canonical_schema: dict) -> None:
        """Completed models training output validates against the schema."""
        tracker = ComponentStatusTracker(str(tmp_path), "autogluon_models_training")
        tracker.set_metadata(display_name="Models Training Status")
        tracker.record("model_selection", "started")
        tracker.record(
            "model_selection",
            "completed",
            metrics={"top_n": 3, "selected_models": ["LightGBM", "XGBoost", "RF"]},
        )
        tracker.record("refit_and_evaluate", "started")
        tracker.record(
            "refit_and_evaluate",
            "completed",
            metrics={"model_count": 3, "eval_metric": "r2"},
        )
        tracker.record("build_leaderboard", "started")
        tracker.record(
            "build_leaderboard",
            "completed",
            metrics={"best_model": "LightGBM_FULL", "model_count": 3},
        )

        _assert_schema_valid(_save_and_load(tracker), canonical_schema)

    def test_running_with_step(self, tmp_path: Path, canonical_schema: dict) -> None:
        """In-progress stage with step and message validates against the schema."""
        tracker = ComponentStatusTracker(str(tmp_path), "autogluon_models_training")
        tracker.set_metadata(display_name="Models Training Status")
        tracker.record(
            "model_selection",
            "running",
            step="model_training",
            message={"level": "info", "text": "Training models..."},
        )

        _assert_schema_valid(_save_and_load(tracker), canonical_schema)

    def test_failed_stage(self, tmp_path: Path, canonical_schema: dict) -> None:
        """Failed component output validates against the schema."""
        tracker = ComponentStatusTracker(str(tmp_path), "autogluon_models_training")
        tracker.set_metadata(display_name="Models Training Status")
        tracker.record("model_selection", "started")
        tracker.record("model_selection", "failed", error="ValueError: fit timeout")

        _assert_schema_valid(_save_and_load(tracker), canonical_schema)

    def test_context_manager_failure(self, tmp_path: Path, canonical_schema: dict) -> None:
        """Output from context manager exception path validates against the schema."""
        with pytest.raises(RuntimeError):
            with ComponentStatusTracker(str(tmp_path), "automl_data_loader") as status:
                status.set_metadata(display_name="Data Loader Status")
                status.record("prepare_data", "started")
                raise RuntimeError("connection lost")

        data = load_component_status(str(tmp_path))
        _assert_schema_valid(data, canonical_schema)

    def test_timeseries_data_loader(self, tmp_path: Path, canonical_schema: dict) -> None:
        """Timeseries data loader output validates against the schema."""
        tracker = ComponentStatusTracker(str(tmp_path), "timeseries_data_loader")
        tracker.set_metadata(display_name="Timeseries Data Loader Status")
        tracker.record("prepare_data", "started")
        tracker.record("prepare_data", "running", metrics={"source": "s3://bucket/ts.csv"})
        tracker.record("prepare_data", "completed", metrics={"rows": 1000})
        tracker.record("split_and_export", "started")
        tracker.record(
            "split_and_export",
            "completed",
            metrics={"test_size": 0.2, "selection_train_size": 0.3},
        )

        _assert_schema_valid(_save_and_load(tracker), canonical_schema)

    def test_timeseries_models_training(self, tmp_path: Path, canonical_schema: dict) -> None:
        """Timeseries models training output validates against the schema."""
        tracker = ComponentStatusTracker(str(tmp_path), "autogluon_timeseries_models_training")
        tracker.set_metadata(display_name="Timeseries Models Training Status")
        tracker.record("model_selection", "started")
        tracker.record(
            "model_selection",
            "completed",
            metrics={"top_n": 3, "selected_models": ["AutoETS", "DeepAR", "PatchTST"]},
        )
        tracker.record("refit_and_evaluate", "started")
        tracker.record(
            "refit_and_evaluate",
            "completed",
            metrics={"model_count": 3, "eval_metric": "mean_absolute_scaled_error"},
        )
        tracker.record("build_leaderboard", "started")
        tracker.record(
            "build_leaderboard",
            "completed",
            metrics={"best_model": "AutoETS_FULL", "model_count": 3},
        )

        _assert_schema_valid(_save_and_load(tracker), canonical_schema)

    def test_flat_keys_rejected_by_schema(self, tmp_path: Path, canonical_schema: dict) -> None:
        """Manually injected flat keys on a stage fail schema validation."""
        from jsonschema import ValidationError, validate

        tracker = ComponentStatusTracker(str(tmp_path), "automl_data_loader")
        tracker.set_metadata(display_name="Data Loader Status")
        tracker.record("prepare_data", "completed")

        data = _save_and_load(tracker)
        data["stages"][0]["rows"] = 100

        with pytest.raises(ValidationError, match="Additional properties are not allowed"):
            validate(instance=data, schema=canonical_schema)
