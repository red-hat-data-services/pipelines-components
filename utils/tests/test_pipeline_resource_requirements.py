"""Unit tests for pipeline executor resource helpers and AutoML/AutoRAG tier matrices.

When a pipeline task's CPU/memory tier changes (for example increasing the default
``speed`` preset training resources), update the constants in ``utils/pipeline_task_resources.py``.
"""

import pytest
from kfp_components.pipelines.data_processing.autorag.documents_indexing_pipeline.pipeline import (
    documents_indexing_pipeline,
)
from kfp_components.pipelines.training.automl.autogluon_tabular_training_pipeline.pipeline import (
    autogluon_tabular_training_pipeline,
)
from kfp_components.pipelines.training.automl.autogluon_timeseries_training_pipeline.pipeline import (
    autogluon_timeseries_training_pipeline,
)
from kfp_components.pipelines.training.autorag.documents_rag_optimization_pipeline.pipeline import (
    documents_rag_optimization_pipeline,
)
from kfp_components.utils.pipeline_task_resources import (
    AUTOML_TABULAR_EXECUTOR_RESOURCES,
    AUTOML_TIMESERIES_EXECUTOR_RESOURCES,
    AUTORAG_INDEXING_EXECUTOR_RESOURCES,
    AUTORAG_OPTIMIZATION_EXECUTOR_RESOURCES,
    TRAINING_BALANCED_RESOURCES,
    TRAINING_SPEED_RESOURCES,
    ExecutorResources,
    assert_executor_resources,
    compile_executor_resources,
    normalize_executor_name,
)


class TestPipelineTaskResourcesHelpers:
    """Tests for compile/assert helpers in pipeline_task_resources."""

    def test_normalize_executor_name_strips_prefix(self):
        """Executor keys drop the ``exec-`` prefix for stable task names."""
        assert normalize_executor_name("exec-automl-data-loader") == "automl-data-loader"

    def test_assert_executor_resources_detects_cpu_change(self):
        """Mismatched CPU requests fail with the task name in the error."""
        actual = {
            "exec-automl-data-loader": ExecutorResources("2", "8Gi", "32", "64Gi"),
        }
        expected = {
            "automl-data-loader": ExecutorResources("4", "8Gi", "32", "64Gi"),
        }
        with pytest.raises(AssertionError, match="automl-data-loader"):
            assert_executor_resources(actual, expected, pipeline_name="test-pipeline")

    def test_assert_executor_resources_allow_extra_executors(self):
        """Partial expected maps can ignore additional executors when allow_extra is set."""
        actual = {
            "exec-automl-data-loader": ExecutorResources("2", "8Gi", "32", "64Gi"),
            "exec-leaderboard-evaluation": ExecutorResources("1", "4Gi", "32", "64Gi"),
        }
        expected = {
            "automl-data-loader": ExecutorResources("2", "8Gi", "32", "64Gi"),
        }
        assert_executor_resources(actual, expected, pipeline_name="test-pipeline", allow_extra=True)


class TestAutoMLPipelineResourceRequirements:
    """AutoML pipelines declare preset-dependent training tiers plus shared loader/leaderboard tiers."""

    def test_tabular_pipeline_executor_resources(self):
        """All tabular pipeline executors match the declared CPU/memory matrix."""
        assert_executor_resources(
            compile_executor_resources(autogluon_tabular_training_pipeline),
            AUTOML_TABULAR_EXECUTOR_RESOURCES,
            pipeline_name="autogluon_tabular_training_pipeline",
        )

    def test_timeseries_pipeline_executor_resources(self):
        """All time series pipeline executors match the declared CPU/memory matrix."""
        assert_executor_resources(
            compile_executor_resources(autogluon_timeseries_training_pipeline),
            AUTOML_TIMESERIES_EXECUTOR_RESOURCES,
            pipeline_name="autogluon_timeseries_training_pipeline",
        )

    def test_default_speed_preset_uses_lower_training_tier(self):
        """Default speed preset branch requests less CPU/memory than balanced."""
        for pipeline_func in (autogluon_tabular_training_pipeline, autogluon_timeseries_training_pipeline):
            actual = compile_executor_resources(pipeline_func)
            speed_keys = [name for name in actual if name.endswith("-2") and "models-training" in name]
            balanced_keys = [name for name in actual if "models-training" in name and not name.endswith("-2")]
            assert len(speed_keys) == 1, f"Expected one speed training executor in {pipeline_func.__name__}"
            assert len(balanced_keys) == 1, f"Expected one balanced training executor in {pipeline_func.__name__}"
            speed = actual[speed_keys[0]]
            balanced = actual[balanced_keys[0]]
            assert speed == TRAINING_SPEED_RESOURCES
            assert balanced == TRAINING_BALANCED_RESOURCES
            assert float(speed.cpu_request) < float(balanced.cpu_request)


class TestAutoRAGPipelineResourceRequirements:
    """AutoRAG pipelines assign workload tiers to data/HPO steps and a lighter tier to leaderboard."""

    def test_rag_optimization_pipeline_executor_resources(self):
        """RAG optimization pipeline sets resources on every component step."""
        assert_executor_resources(
            compile_executor_resources(documents_rag_optimization_pipeline),
            AUTORAG_OPTIMIZATION_EXECUTOR_RESOURCES,
            pipeline_name="documents_rag_optimization_pipeline",
        )

    def test_documents_indexing_pipeline_executor_resources(self):
        """Documents indexing pipeline sets workload-tier resources on all three steps."""
        assert_executor_resources(
            compile_executor_resources(documents_indexing_pipeline),
            AUTORAG_INDEXING_EXECUTOR_RESOURCES,
            pipeline_name="documents_indexing_pipeline",
        )

    def test_indexing_pipeline_shared_components_match_rag_optimization(self):
        """documents-discovery and text-extraction use the same tier in both AutoRAG pipelines."""
        rag_resources = compile_executor_resources(documents_rag_optimization_pipeline)
        indexing_resources = compile_executor_resources(documents_indexing_pipeline)

        rag_by_task = {normalize_executor_name(name): res for name, res in rag_resources.items()}
        indexing_by_task = {normalize_executor_name(name): res for name, res in indexing_resources.items()}

        for task_name in ("documents-discovery", "text-extraction"):
            assert task_name in rag_by_task, f"{task_name} missing from RAG optimization pipeline"
            assert task_name in indexing_by_task, f"{task_name} missing from indexing pipeline"
            assert rag_by_task[task_name] == indexing_by_task[task_name], (
                f"{task_name} resources differ between pipelines: "
                f"rag={rag_by_task[task_name]}, indexing={indexing_by_task[task_name]}"
            )
