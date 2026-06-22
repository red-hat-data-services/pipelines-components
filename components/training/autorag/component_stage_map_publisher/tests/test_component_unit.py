"""Unit tests for AutoRAG publish_component_stage_map."""

import json
from pathlib import Path
from unittest import mock

import pytest

from ..component import publish_component_stage_map

PIPELINE_RAG_OPTIMIZATION = "documents-rag-optimization-pipeline"


@pytest.fixture
def component_stage_map_artifact(tmp_path):
    """Mock KFP output artifact for component_stage_map."""
    artifact = mock.MagicMock()
    artifact.path = str(tmp_path / "component_stage_map")
    artifact.metadata = {}
    return artifact


class TestPublishComponentStageMap:
    """Unit tests for the AutoRAG component stage map publisher."""

    def test_component_is_callable(self):
        """Component is importable and exposes python_func."""
        assert callable(publish_component_stage_map)
        assert hasattr(publish_component_stage_map, "python_func")

    def test_publishes_stage_map_from_template(self, component_stage_map_artifact):
        """Publish AutoRAG template as component_stage_map.json with runtime fields."""
        publish_component_stage_map.python_func(
            pipeline_id=PIPELINE_RAG_OPTIMIZATION,
            run_id="run-abc",
            component_stage_map=component_stage_map_artifact,
        )
        output_file = Path(component_stage_map_artifact.path) / "component_stage_map.json"
        assert output_file.is_file()
        document = json.loads(output_file.read_text(encoding="utf-8"))
        assert document["pipeline_id"] == PIPELINE_RAG_OPTIMIZATION
        assert document["kfp_run_id"] == "run-abc"
        assert "published_at" in document
        assert len(document["components"]) >= 1
        assert component_stage_map_artifact.metadata["display_name"] == "Component Stage Map"
        assert component_stage_map_artifact.metadata["pipeline_id"] == PIPELINE_RAG_OPTIMIZATION
        assert component_stage_map_artifact.metadata["component_count"] == len(document["components"])

    def test_rejects_empty_pipeline_id(self, component_stage_map_artifact):
        """Reject blank pipeline_id."""
        with pytest.raises(ValueError, match="pipeline_id"):
            publish_component_stage_map.python_func(
                pipeline_id="  ",
                run_id="run-1",
                component_stage_map=component_stage_map_artifact,
            )

    def test_rejects_empty_run_id(self, component_stage_map_artifact):
        """Reject blank run_id."""
        with pytest.raises(ValueError, match="run_id"):
            publish_component_stage_map.python_func(
                pipeline_id=PIPELINE_RAG_OPTIMIZATION,
                run_id="",
                component_stage_map=component_stage_map_artifact,
            )

    @pytest.mark.parametrize("invalid_pipeline_id", ["foo/bar", "foo\\bar", "../../../etc/passwd"])
    def test_rejects_path_traversal_pipeline_id(self, component_stage_map_artifact, invalid_pipeline_id):
        """Reject pipeline_id values that could be used for path traversal."""
        with pytest.raises(ValueError, match="must be a simple identifier"):
            publish_component_stage_map.python_func(
                pipeline_id=invalid_pipeline_id,
                run_id="run-1",
                component_stage_map=component_stage_map_artifact,
            )

    def test_rejects_empty_components_list(self, component_stage_map_artifact, tmp_path):
        """Reject templates whose components list is empty."""
        templates_root = tmp_path / "run_status_templates"
        pipelines_dir = templates_root / "pipelines"
        pipelines_dir.mkdir(parents=True)
        (pipelines_dir / "empty-pipeline.json").write_text(
            json.dumps({"pipeline_id": "empty-pipeline", "components": []}),
            encoding="utf-8",
        )

        with pytest.raises(FileNotFoundError, match="empty-pipeline"):
            publish_component_stage_map.python_func(
                pipeline_id="empty-pipeline",
                run_id="run-1",
                component_stage_map=component_stage_map_artifact,
                embedded_artifact=type("Embedded", (), {"path": str(templates_root)})(),
            )

    def test_unknown_pipeline_id_raises(self, component_stage_map_artifact):
        """Raise when template has no components for pipeline_id."""
        with pytest.raises(FileNotFoundError, match="nonexistent-pipeline"):
            publish_component_stage_map.python_func(
                pipeline_id="nonexistent-pipeline",
                run_id="run-1",
                component_stage_map=component_stage_map_artifact,
            )
