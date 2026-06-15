"""Tests for AutoRAG component-local status tracking."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from kfp_components.components.training.autorag.shared.component_status import (
    COMPONENT_STATUS_FILENAME,
    ComponentStatusEncoder,
    ComponentStatusTracker,
    bootstrap_status_tracker,
    component_status_tracker,
    load_component_status,
    load_embedded_component_status_module,
)


class TestComponentStatusTracker:
    """Tests for ComponentStatusTracker."""

    def test_record_and_save(self, tmp_path: Path) -> None:
        """save() writes stages and component_id to component_status.json."""
        tracker = ComponentStatusTracker(str(tmp_path), "test_data_loader")
        tracker.record("validate_inputs", "started", rows=5)
        tracker.record("validate_inputs", "completed")
        tracker.save()

        data = json.loads((tmp_path / COMPONENT_STATUS_FILENAME).read_text(encoding="utf-8"))
        assert data["component_id"] == "test_data_loader"
        assert len(data["stages"]) == 1
        assert data["stages"][0]["status"] == "completed"
        assert data["stages"][0]["rows"] == 5

    def test_disabled_tracker_skips_save(self, tmp_path: Path) -> None:
        """When artifact_path is None, save() is a no-op."""
        tracker = ComponentStatusTracker(None, "test_data_loader")
        tracker.record("validate_inputs", "completed")
        tracker.save()
        assert not (tmp_path / COMPONENT_STATUS_FILENAME).exists()

    def test_component_status_tracker_from_none(self, tmp_path: Path) -> None:
        """component_status_tracker() accepts a missing artifact for unit tests."""
        tracker = component_status_tracker(None, "documents_discovery")
        tracker.record("validate_inputs", "completed")
        tracker.save()
        assert not (tmp_path / COMPONENT_STATUS_FILENAME).exists()

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
        tracker.record("run_optimization", "started")
        tracker.record("run_optimization", "completed", steps=steps)
        tracker.save()

        data = load_component_status(str(tmp_path))
        assert data["stages"][-1]["steps"] == steps

    def test_stage_skips_auto_complete_when_completed_inside_block(self, tmp_path: Path) -> None:
        """stage() does not overwrite a completed record written inside the block."""
        tracker = ComponentStatusTracker(str(tmp_path), "rag_templates_optimization")
        with tracker.stage("run_optimization", steps=["chunking"]):
            tracker.record("run_optimization", "completed", max_rag_patterns=8)
        tracker.save()

        data = load_component_status(str(tmp_path))
        run_stage = next(stage for stage in data["stages"] if stage["id"] == "run_optimization")
        assert run_stage["status"] == "completed"
        assert run_stage["max_rag_patterns"] == 8


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
        status.record("validate_inputs", "completed")
        status.save()
        assert (tmp_path / COMPONENT_STATUS_FILENAME).is_file()

    def test_load_embedded_module_from_file_path(self) -> None:
        """load_embedded_component_status_module accepts a file embedded artifact path."""
        module_path = Path(__file__).resolve().parents[1] / "component_status.py"
        embedded = type("Embedded", (), {"path": str(module_path)})()
        module = load_embedded_component_status_module(embedded)
        assert hasattr(module, "bootstrap_status_tracker")


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
