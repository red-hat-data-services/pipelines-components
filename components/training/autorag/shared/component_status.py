"""Component-local status tracking for AutoRAG pipelines.

Each component publishes its own stage progress to an artifact. No workspace required.
The dashboard aggregates component statuses to show overall pipeline progress.

Usage:
    from kfp_components.components.training.autorag.shared.component_status import (
        component_status_tracker,
    )

    status = component_status_tracker(component_status, "test_data_loader")
    with status:
        with status.stage("download_and_sample"):
            ...
"""

from __future__ import annotations

import base64
import json
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any

logger = logging.getLogger(__name__)


class ComponentStatusEncoder(json.JSONEncoder):
    """Custom JSON encoder for component status data.

    Handles types commonly found in component metadata that aren't JSON-serializable
    by default: datetime, Path, bytes, sets, and other objects.
    """

    def default(self, obj: Any) -> Any:
        """Convert non-serializable objects to JSON-compatible types."""
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, bytes):
            return base64.b64encode(obj).decode("ascii")
        if isinstance(obj, set):
            return sorted(obj, key=str)
        return super().default(obj)


COMPONENT_STATUS_FILENAME = "component_status.json"

STATUS_STARTED = "started"
STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"


class ComponentStatusTracker:
    """Track stage-level progress within a single AutoRAG component."""

    def __init__(self, artifact_path: str | None, component_id: str) -> None:
        """Initialize the status tracker.

        Args:
            artifact_path: Path to the KFP artifact directory where status.json will be written.
                When ``None``, tracking is disabled (e.g. direct unit-test invocations without a mock artifact).
            component_id: Unique component identifier (e.g., "test_data_loader").
        """
        self._enabled = artifact_path is not None
        self.artifact_path = Path(artifact_path) if self._enabled else Path(".")
        self.component_id = component_id
        self.stages: list[dict[str, Any]] = []
        self.started_at = self._utc_now_iso()
        self.metadata: dict[str, Any] = {}

    @staticmethod
    def _utc_now_iso() -> str:
        """Return current UTC timestamp in ISO format."""
        return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    def record(self, stage_id: str, status: str, **metadata: Any) -> None:
        """Record or update a stage's status."""
        existing_idx = next((i for i, s in enumerate(self.stages) if s["id"] == stage_id), None)

        stage_data = {
            "id": stage_id,
            "status": status,
            "timestamp": self._utc_now_iso(),
            **metadata,
        }

        if existing_idx is not None:
            self.stages[existing_idx].update(stage_data)
        else:
            self.stages.append(stage_data)

        logger.info(
            "COMPONENT_STATUS component=%s stage=%s status=%s %s",
            self.component_id,
            stage_id,
            status,
            " ".join(f"{k}={v}" for k, v in metadata.items()),
        )

    def set_metadata(self, **metadata: Any) -> None:
        """Set component-level metadata."""
        self.metadata.update(metadata)

    def save(self) -> None:
        """Write the final status to the artifact."""
        if not self._enabled:
            return

        self.artifact_path.mkdir(parents=True, exist_ok=True)

        data = {
            "component_id": self.component_id,
            "started_at": self.started_at,
            "completed_at": self._utc_now_iso(),
            "stages": self.stages,
            "metadata": self.metadata,
        }

        output_file = self.artifact_path / COMPONENT_STATUS_FILENAME
        with output_file.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, cls=ComponentStatusEncoder)

        logger.info(
            "COMPONENT_STATUS component=%s saved status with %d stages to %s",
            self.component_id,
            len(self.stages),
            output_file,
        )

    def save_best_effort(self) -> None:
        """Write status to the artifact, logging instead of raising on I/O errors."""
        try:
            self.save()
        except Exception:
            logger.exception(
                "Failed to save component status for %s",
                self.component_id,
            )

    def mark_active_failed(self, error: str | BaseException) -> None:
        """Mark the in-progress stage as failed, or the last open stage if none is active."""
        # Handle empty str(exc) by including exception type
        if isinstance(error, BaseException):
            error_msg = f"{type(error).__name__}: {error}" if str(error) else type(error).__name__
        else:
            error_msg = error

        active_statuses = (STATUS_STARTED, STATUS_RUNNING)
        for stage in reversed(self.stages):
            if stage.get("status") in active_statuses:
                self.record(stage["id"], STATUS_FAILED, error=error_msg)
                return

        if self.stages and self.stages[-1].get("status") != STATUS_COMPLETED:
            self.record(self.stages[-1]["id"], STATUS_FAILED, error=error_msg)
            return

        self.set_metadata(status=STATUS_FAILED, error=error_msg)

    @contextmanager
    def stage(self, stage_id: str, **start_metadata: Any) -> Iterator[None]:
        """Record stage started/completed, or failed when an exception escapes the block."""
        self.record(stage_id, STATUS_STARTED, **start_metadata)
        try:
            yield
        except BaseException as exc:
            # Use exception-aware error formatting
            error_msg = f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
            self.record(stage_id, STATUS_FAILED, error=error_msg)
            raise
        else:
            latest = next((s for s in reversed(self.stages) if s["id"] == stage_id), None)
            if latest is None or latest.get("status") not in (STATUS_COMPLETED, STATUS_FAILED):
                self.record(stage_id, STATUS_COMPLETED)

    def __enter__(self) -> ComponentStatusTracker:
        """Enter context: return this tracker."""
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: Any) -> bool:
        """On exit, mark active stage failed and save status best-effort."""
        if exc is not None:
            self.mark_active_failed(exc)
        self.save_best_effort()
        return False


def _resolve_embedded_module_path(embedded_artifact: Any) -> Path:
    """Return the path to embedded ``component_status.py`` after validating the artifact."""
    if embedded_artifact is None:
        raise ValueError(
            "embedded_artifact is required for component status tracking. "
            "Injected by KFP when using embedded_artifact_path."
        )
    if not getattr(embedded_artifact, "path", None):
        raise ValueError("embedded_artifact.path is missing or empty")

    embedded_path = Path(embedded_artifact.path)
    if embedded_path.is_file():
        module_path = embedded_path
    elif embedded_path.is_dir():
        module_path = embedded_path / "component_status.py"
    else:
        raise ValueError(f"Invalid embedded_artifact.path: {embedded_path}")
    if not module_path.is_file():
        raise ValueError(f"Embedded component_status module not found at {module_path}")
    return module_path


def resolve_embedded_import_root(embedded_artifact: Any) -> Path:
    """Return the directory containing embedded shared assets (e.g. script templates)."""
    embedded_path = Path(embedded_artifact.path)
    if embedded_path.is_file():
        return embedded_path.parent
    if embedded_path.is_dir():
        return embedded_path
    raise ValueError(f"Invalid embedded_artifact.path: {embedded_path}")


def load_embedded_component_status_module(embedded_artifact: Any) -> ModuleType:
    """Load ``component_status`` from a KFP embedded artifact path via importlib."""
    import importlib.util

    module_path = _resolve_embedded_module_path(embedded_artifact)
    spec = importlib.util.spec_from_file_location("_autorag_component_status", module_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load embedded module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tracker_from_embedded(
    embedded_artifact: Any,
    component_status: Any,
    component_id: str,
) -> ComponentStatusTracker:
    """Validate embedded artifact path and return a configured tracker."""
    _resolve_embedded_module_path(embedded_artifact)
    return component_status_tracker(component_status, component_id)


def bootstrap_status_tracker(
    embedded_artifact: Any,
    component_status: Any,
    component_id: str,
) -> ComponentStatusTracker:
    """Load this module from ``embedded_artifact`` and return a configured tracker."""
    status_module = load_embedded_component_status_module(embedded_artifact)
    return status_module.tracker_from_embedded(embedded_artifact, component_status, component_id)


def component_status_tracker(component_status: Any, component_id: str) -> ComponentStatusTracker:
    """Build a tracker from an optional KFP ``component_status`` output artifact."""
    artifact_path = component_status.path if component_status is not None else None
    return ComponentStatusTracker(artifact_path, component_id)


def load_component_status(artifact_path: str) -> dict[str, Any]:
    """Load component status from an artifact directory.

    Returns an empty dict if the file doesn't exist or is corrupted.
    """
    status_file = Path(artifact_path) / COMPONENT_STATUS_FILENAME
    if not status_file.exists():
        return {}

    try:
        with status_file.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to load status from %s: %s", status_file, e)
        return {}
