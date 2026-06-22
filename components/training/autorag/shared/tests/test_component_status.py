"""Tests for AutoRAG component-local status tracking."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from kfp_components.components.training.autorag.shared.component_status import (
    COMPONENT_STATUS_FILENAME,
    ComponentStatusEncoder,
    ComponentStatusTracker,
    NullComponentStatusTracker,
    bootstrap_status_tracker,
    load_component_status,
    load_embedded_component_status_module,
)


class TestComponentStatusTracker:
    """Tests for ComponentStatusTracker."""

    def test_record_and_save(self, tmp_path: Path) -> None:
        """save() writes stages and component_id to component_status.json."""
        tracker = ComponentStatusTracker(str(tmp_path), "test_data_loader")
        tracker.record("load_benchmark", "started", rows=5)
        tracker.record("load_benchmark", "completed")
        tracker.save()

        data = json.loads((tmp_path / COMPONENT_STATUS_FILENAME).read_text(encoding="utf-8"))
        assert data["component_id"] == "test_data_loader"
        assert len(data["stages"]) == 1
        assert data["stages"][0]["status"] == "completed"
        assert data["stages"][0]["rows"] == 5

    def test_context_manager_marks_failed_and_saves(self, tmp_path: Path) -> None:
        """Context manager marks active stage failed and persists status on exception."""
        with pytest.raises(RuntimeError, match="boom"):
            with ComponentStatusTracker(str(tmp_path), "text_extraction") as status:
                status.record("extract_documents", "started")
                raise RuntimeError("boom")

        data = load_component_status(str(tmp_path))
        assert data["stages"][-1]["status"] == "failed"
        assert "boom" in data["stages"][-1]["error"]

    def test_record_completed_with_steps(self, tmp_path: Path) -> None:
        """Completed stages can include manifest step ids for dashboard display."""
        tracker = ComponentStatusTracker(str(tmp_path), "rag_templates_optimization")
        steps = ["chunking", "embedding", "retrieval", "generation", "evaluation"]
        tracker.record("optimize_templates", "started")
        tracker.record("optimize_templates", "completed", steps=steps)
        tracker.save()

        data = load_component_status(str(tmp_path))
        assert data["stages"][-1]["steps"] == steps

    def test_stage_skips_auto_complete_when_completed_inside_block(self, tmp_path: Path) -> None:
        """stage() does not overwrite a completed record written inside the block."""
        tracker = ComponentStatusTracker(str(tmp_path), "rag_templates_optimization")
        with tracker.stage("optimize_templates", steps=["chunking"]):
            tracker.record("optimize_templates", "completed", max_rag_patterns=8)
        tracker.save()

        data = load_component_status(str(tmp_path))
        run_stage = next(stage for stage in data["stages"] if stage["id"] == "optimize_templates")
        assert run_stage["status"] == "completed"
        assert run_stage["max_rag_patterns"] == 8

    def test_utc_now_z_ends_with_z(self) -> None:
        """Timestamps use UTC ISO-8601 with Z suffix."""
        from kfp_components.components.training.autorag.shared.component_status import utc_now_z

        assert utc_now_z().endswith("Z")

    def test_component_status_json_schema_compliance(self, tmp_path: Path) -> None:
        """component_status.json follows expected schema for dashboard consumption."""
        from datetime import datetime

        tracker = ComponentStatusTracker(str(tmp_path), "test_component")
        tracker.record("stage1", "started")
        tracker.record("stage1", "completed", rows=100, files=5)
        tracker.record("stage2", "started", input_size_mb=50)
        tracker.record("stage2", "completed", output_size_mb=45)
        tracker.save()

        data = json.loads((tmp_path / COMPONENT_STATUS_FILENAME).read_text(encoding="utf-8"))

        # Top-level structure
        required_fields = {"component_id", "started_at", "completed_at", "stages", "metadata"}
        assert set(data.keys()) >= required_fields, f"Missing fields: {required_fields - set(data.keys())}"
        assert data["component_id"] == "test_component"

        # ISO-8601 timestamps with Z suffix
        assert data["started_at"].endswith("Z"), "started_at should end with Z"
        assert data["completed_at"].endswith("Z"), "completed_at should end with Z"

        # Verify timestamps are valid ISO-8601
        datetime.fromisoformat(data["started_at"].replace("Z", "+00:00"))
        datetime.fromisoformat(data["completed_at"].replace("Z", "+00:00"))

        # Stages array structure
        assert isinstance(data["stages"], list), "stages should be a list"
        assert len(data["stages"]) == 2, "Should have 2 stages"

        # Verify each stage has required fields
        for stage in data["stages"]:
            assert "id" in stage, "Each stage must have an id"
            assert "status" in stage, "Each stage must have a status"
            assert "timestamp" in stage, "Each stage must have a timestamp"
            assert stage["timestamp"].endswith("Z"), "Stage timestamps should end with Z"

        # Verify stage 1 metadata
        stage1 = next(s for s in data["stages"] if s["id"] == "stage1")
        assert stage1["status"] == "completed"
        assert stage1["rows"] == 100
        assert stage1["files"] == 5

        # Verify stage 2 metadata
        stage2 = next(s for s in data["stages"] if s["id"] == "stage2")
        assert stage2["status"] == "completed"
        assert stage2["input_size_mb"] == 50
        assert stage2["output_size_mb"] == 45


class TestComponentStatusEncoder:
    """Tests for JSON encoding of status metadata values."""

    def test_encodes_datetime_path_bytes_and_set(self) -> None:
        """Known non-JSON types are converted for serialization."""
        encoder = ComponentStatusEncoder()
        assert encoder.default(Path("/tmp/out")) == "/tmp/out"
        assert encoder.default(b"abc") == "YWJj"
        assert encoder.default({1, 2}) == [1, 2]

    def test_unknown_type_raises_type_error(self) -> None:
        """Unsupported metadata types fail fast instead of being stringified."""
        encoder = ComponentStatusEncoder()
        with pytest.raises(TypeError):
            encoder.default(object())


class TestEmbeddedStatusBootstrap:
    """Tests for embedded-artifact loader helpers."""

    def test_bootstrap_status_tracker_from_shared_dir(self, tmp_path: Path) -> None:
        """bootstrap_status_tracker loads from a directory embedded artifact path."""
        shared_dir = Path(__file__).resolve().parents[1]
        embedded = type("Embedded", (), {"path": str(shared_dir)})()
        status = bootstrap_status_tracker(embedded, type("Status", (), {"path": str(tmp_path)})(), "test_data_loader")
        status.record("load_benchmark", "completed")
        status.save()
        assert (tmp_path / COMPONENT_STATUS_FILENAME).is_file()

    def test_load_embedded_module_from_file_path(self) -> None:
        """load_embedded_component_status_module accepts a file embedded artifact path."""
        module_path = Path(__file__).resolve().parents[1] / "component_status.py"
        embedded = type("Embedded", (), {"path": str(module_path)})()
        module = load_embedded_component_status_module(embedded)
        assert hasattr(module, "bootstrap_status_tracker")

    def test_bootstrap_status_tracker_returns_noop_when_component_status_is_none(self) -> None:
        """Notebook-style invocations without component_status use a no-op tracker."""
        embedded = type("Embedded", (), {"path": str(Path(__file__).resolve().parents[1])})()
        status = bootstrap_status_tracker(embedded, None, "documents_discovery")
        assert isinstance(status, NullComponentStatusTracker)
        with status:
            with status.stage("discover_documents"):
                pass


class TestNullComponentStatusTracker:
    """Tests for no-op status tracker used in notebook execution."""

    def test_null_tracker_propagates_exceptions(self) -> None:
        """Exceptions escape the context manager (not suppressed)."""
        from kfp_components.components.training.autorag.shared.component_status import (
            null_component_status_tracker,
        )

        status = null_component_status_tracker()
        with pytest.raises(ValueError, match="test error"):
            with status:
                raise ValueError("test error")

    def test_null_tracker_creates_no_files(self, tmp_path: Path) -> None:
        """No artifacts written when using null tracker."""
        status = NullComponentStatusTracker()
        with status:
            status.record("stage1", "started")
            status.set_metadata(key="value")
            with status.stage("stage2"):
                pass

        # Verify no files created in tmp_path
        assert list(tmp_path.iterdir()) == []

    def test_null_tracker_stage_propagates_exceptions(self) -> None:
        """stage() context manager doesn't suppress exceptions."""
        from kfp_components.components.training.autorag.shared.component_status import (
            null_component_status_tracker,
        )

        status = null_component_status_tracker()
        with pytest.raises(RuntimeError, match="boom"):
            with status.stage("test_stage"):
                raise RuntimeError("boom")


class TestComponentStatusTrackerStage:
    """Additional stage() behaviour tests."""

    def test_stage_marks_failed_on_base_exception_subclass(self, tmp_path: Path) -> None:
        """stage() records failed when a BaseException subclass escapes the block."""
        with pytest.raises(KeyboardInterrupt):
            with ComponentStatusTracker(str(tmp_path), "text_extraction") as tracker:
                with tracker.stage("extract_documents"):
                    raise KeyboardInterrupt

        data = load_component_status(str(tmp_path))
        assert data["stages"][-1]["status"] == "failed"
        assert data["stages"][-1]["error"] == "KeyboardInterrupt"
