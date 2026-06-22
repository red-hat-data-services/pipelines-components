"""Unit tests for documents indexing pipeline executor resource tiers."""

from kfp_components.utils.pipeline_task_resources import (
    assert_executor_resources,
    compile_executor_resources,
)

from ..pipeline import documents_indexing_pipeline
from .pipeline_resource_expectations import AUTORAG_INDEXING_EXECUTOR_RESOURCES


class TestDocumentsIndexingPipelineResourceRequirements:
    """Documents indexing pipeline sets workload-tier resources on all three steps."""

    def test_documents_indexing_pipeline_executor_resources(self):
        """Documents indexing pipeline sets workload-tier resources on all three steps."""
        assert_executor_resources(
            compile_executor_resources(documents_indexing_pipeline),
            AUTORAG_INDEXING_EXECUTOR_RESOURCES,
            pipeline_name="documents_indexing_pipeline",
        )
