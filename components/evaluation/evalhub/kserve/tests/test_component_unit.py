"""Unit tests for the evalhub_kserve component."""

from ..component import evalhub_evaluator_kserve


class TestEvalhubKserveUnitTests:
    """Unit tests for component logic."""

    def test_component_function_exists(self):
        """Test that the component function is properly imported."""
        assert callable(evalhub_evaluator_kserve)
        assert hasattr(evalhub_evaluator_kserve, "python_func")

    def test_component_has_expected_parameters(self):
        """Test that the component has expected input parameters."""
        import inspect

        sig = inspect.signature(evalhub_evaluator_kserve.python_func)
        params = list(sig.parameters.keys())

        expected_params = [
            "output_metrics",
            "output_results",
            "evalhub_url",
            "benchmarks",
            "collection_id",
            "pvc_mount_path",
            "model_artifact",
            "model_path",
            "evalhub_tenant",
            "evalhub_auth_token",
            "evalhub_model_name",
            "base_model_name",
            "evalhub_job_name",
            "evalhub_timeout",
            "evalhub_poll_interval",
            "mlflow_experiment_name",
            "gpu_count",
            "memory",
            "cpu",
            "runtime_image",
            "trust_remote_code",
            "verify_tls",
            "isvc_ready_timeout",
        ]

        for param in expected_params:
            assert param in params, f"Expected parameter '{param}' not found in component"

    def test_component_default_values(self):
        """Test that the component has expected default values."""
        import inspect

        sig = inspect.signature(evalhub_evaluator_kserve.python_func)
        params = sig.parameters

        assert params["evalhub_model_name"].default == "finetuned-model"
        assert params["evalhub_job_name"].default == "pipeline-eval"
        assert params["evalhub_timeout"].default == 7200
        assert params["evalhub_poll_interval"].default == 30
        assert params["gpu_count"].default == 1
        assert params["memory"].default == "8Gi"
        assert params["cpu"].default == "2"
        assert params["isvc_ready_timeout"].default == 600
        assert params["mlflow_experiment_name"].default == ""
        assert params["collection_id"].default == ""

    def test_component_supports_model_sources(self):
        """Test that the component supports both model_path and model_artifact."""
        import inspect

        sig = inspect.signature(evalhub_evaluator_kserve.python_func)
        params = sig.parameters

        assert params["model_path"].default is None
        assert "model_artifact" in params

    def test_component_supports_benchmark_and_collection(self):
        """Test that both benchmark list and collection ID are accepted."""
        import inspect

        sig = inspect.signature(evalhub_evaluator_kserve.python_func)
        params = sig.parameters

        assert params["benchmarks"].default == []
        assert params["collection_id"].default == ""

    def test_component_kserve_parameters(self):
        """Test that KServe-specific parameters are present."""
        import inspect

        sig = inspect.signature(evalhub_evaluator_kserve.python_func)
        params = list(sig.parameters.keys())

        kserve_params = ["gpu_count", "memory", "cpu", "runtime_image", "isvc_ready_timeout"]
        for param in kserve_params:
            assert param in params, f"KServe parameter '{param}' not found in component"

    def test_component_runtime_image_default(self):
        """Test that the default runtime image is the RHOAI vLLM image."""
        import inspect

        sig = inspect.signature(evalhub_evaluator_kserve.python_func)
        runtime_image = sig.parameters["runtime_image"].default

        assert "registry.redhat.io/rhaii/vllm-cuda-rhel9" in runtime_image
        assert "@sha256:" in runtime_image

    def test_component_docstring(self):
        """Test that the component has a meaningful docstring."""
        docstring = evalhub_evaluator_kserve.python_func.__doc__

        assert "eval hub" in docstring.lower() or "kserve" in docstring.lower()
        assert "inferenceservice" in docstring.lower() or "servingruntime" in docstring.lower()
