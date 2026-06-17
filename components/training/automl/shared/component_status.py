"""Component-local status tracking for AutoML pipelines.

Each component publishes its own stage progress to an artifact. No workspace required.
The dashboard aggregates component statuses to show overall pipeline progress.

Usage:
    with ComponentStatusTracker(artifact.path, "autogluon_models_training") as status:
        status.record("load_data", "started")
        status.record("load_data", "completed", rows=1000)
        with status.stage("model_selection"):
            ...  # marks started/completed; marks failed on exception
    # context exit saves best-effort and marks active stage failed on error
"""

from __future__ import annotations

import base64
import json
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

COMPONENT_STATUS_FILENAME = "component_status.json"

STATUS_STARTED = "started"
STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"


class ComponentStatusEncoder(json.JSONEncoder):
    """JSON encoder for component status metadata (Path, bytes, set, datetime)."""

    def default(self, obj: Any) -> Any:
        """Convert non-serializable objects to JSON-compatible types."""
        if isinstance(obj, datetime):
            return obj.isoformat().replace("+00:00", "Z") if obj.tzinfo is not None else obj.isoformat()
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, bytes):
            return base64.b64encode(obj).decode("ascii")
        if isinstance(obj, set):
            return sorted(obj, key=str)
        return super().default(obj)


def utc_now_z() -> str:
    """Return current UTC time as an ISO-8601 string with ``Z`` suffix."""
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


class ComponentStatusTracker:
    """Track stage-level progress within a single component.

    Publishes component-local status to an artifact without requiring workspace.
    Each component independently tracks its stages and metadata.
    """

    def __init__(self, artifact_path: str | None, component_id: str) -> None:
        """Initialize the status tracker.

        Args:
            artifact_path: Path to the KFP artifact directory where status.json will be written.
                When ``None``, tracking is disabled (e.g. unit tests without a mock artifact).
            component_id: Unique component identifier (e.g., "autogluon_models_training").
        """
        self._enabled = artifact_path is not None
        self.artifact_path = Path(artifact_path) if self._enabled else Path(".")
        self.component_id = component_id
        self.stages: list[dict[str, Any]] = []
        self.started_at = utc_now_z()
        self.metadata: dict[str, Any] = {}

    def record(self, stage_id: str, status: str, **metadata: Any) -> None:
        """Record or update a stage's status.

        If the stage already exists, it will be updated. Otherwise, a new stage is appended.

        Args:
            stage_id: Stage identifier (e.g., "load_data", "model_selection").
            status: Stage status ("started", "running", "completed", "failed").
            **metadata: Additional stage data (e.g., rows=1000, steps=["step1", "step2"]).

        Example:
            status.record("load_data", "completed", rows=1000, duration_seconds=5.2)
            status.record("model_selection", "completed",
                         steps=["feature_eng", "training", "stacking"],
                         models_trained=15)
        """
        existing_idx = next((i for i, s in enumerate(self.stages) if s["id"] == stage_id), None)

        stage_data = {
            "id": stage_id,
            "status": status,
            "timestamp": utc_now_z(),
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
        """Set component-level metadata.

        Args:
            **metadata: Key-value pairs to store at component level.

        Example:
            status.set_metadata(total_training_time_seconds=3600, models_produced=5)
        """
        self.metadata.update(metadata)

    def save(self) -> None:
        """Write the final status to the artifact.

        Creates the artifact directory if needed and writes component_status.json
        with all recorded stages and metadata.
        """
        if not self._enabled:
            return

        self.artifact_path.mkdir(parents=True, exist_ok=True)

        data = {
            "component_id": self.component_id,
            "started_at": self.started_at,
            "completed_at": utc_now_z(),
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


def load_component_status(artifact_path: str) -> dict[str, Any]:
    """Load component status from an artifact.

    Utility function for dashboards/monitoring to read component status.

    Args:
        artifact_path: Path to the artifact directory containing component_status.json.

    Returns:
        Dict containing component_id, started_at, completed_at, stages, and metadata.
        Returns empty dict if file doesn't exist or is unreadable.

    Example:
        status = load_component_status("/path/to/artifact")
        print(f"Component {status['component_id']} completed {len(status['stages'])} stages")
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
