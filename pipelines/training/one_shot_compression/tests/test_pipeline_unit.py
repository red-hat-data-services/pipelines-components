"""Tests for the one_shot_compression pipeline."""

import tempfile

from kfp.compiler import Compiler
from kfp.dsl import graph_component

from ..pipeline import one_shot_compression


class TestOneShotCompressionUnitTests:
    """Unit tests for pipeline logic."""

    def test_pipeline_function_exists(self):
        """Test that the pipeline is properly imported as a GraphComponent."""
        assert callable(one_shot_compression)
        assert isinstance(one_shot_compression, graph_component.GraphComponent)

    def test_pipeline_name(self):
        """Test that the pipeline has the expected name."""
        assert one_shot_compression.component_spec.name == "one-shot-compression"

    def test_pipeline_has_expected_inputs(self):
        """Test that all expected input parameters are declared."""
        input_keys = set(one_shot_compression.component_spec.inputs.keys())
        expected = {
            "model_id",
            "dataset_id",
            "dataset_split",
            "quantization_scheme",
            "quantization_ignore_list",
            "cpu_requests",
            "memory_request",
            "accelerator_type",
            "accelerator_limit",
            "model_registry_address",
            "model_registry_author",
            "model_registry_model_name",
            "model_registry_model_version",
            "model_registry_opt_description",
            "model_registry_opt_format_name",
            "model_registry_opt_port",
        }
        assert expected == input_keys

    def test_pipeline_contains_expected_tasks(self):
        """Test that the compiled pipeline DAG contains both tasks."""
        tasks = set(one_shot_compression.pipeline_spec.root.dag.tasks.keys())
        assert "one-shot-model-compressor" in tasks
        assert "kubeflow-model-registry" in tasks

    def test_model_registry_depends_on_compressor(self):
        """Test that the model registry task runs after compression."""
        dag_tasks = one_shot_compression.pipeline_spec.root.dag.tasks
        registry_task = dag_tasks["kubeflow-model-registry"]
        assert "one-shot-model-compressor" in registry_task.dependent_tasks

    def test_pipeline_compiles_to_yaml(self):
        """Test that the pipeline compiles to valid IR YAML without errors."""
        with tempfile.NamedTemporaryFile(suffix=".yaml") as f:
            Compiler().compile(one_shot_compression, package_path=f.name)
            f.seek(0)
            content = f.read()
            assert len(content) > 0
            assert b"pipelineInfo" in content
            assert b"one-shot-compression" in content
