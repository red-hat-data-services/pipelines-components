"""Tests for OSFT Eval Hub pipeline."""

from kfp import compiler

from ..pipeline import osft_pipeline_evalhub


class TestOsftEvalhubPipeline:
    """Basic tests for OSFT Eval Hub pipeline."""

    def test_pipeline_function_exists(self):
        """Test that the pipeline function is properly defined."""
        assert callable(osft_pipeline_evalhub)

    def test_pipeline_compiles(self, tmp_path):
        """Test that the pipeline compiles successfully."""
        output_path = tmp_path / "pipeline.yaml"
        compiler.Compiler().compile(
            pipeline_func=osft_pipeline_evalhub,
            package_path=str(output_path),
        )
        assert output_path.exists()
