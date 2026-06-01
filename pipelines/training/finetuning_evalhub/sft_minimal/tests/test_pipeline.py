"""Tests for SFT Minimal Eval Hub pipeline."""

from kfp import compiler

from ..pipeline import sft_pipeline_evalhub_easy


class TestSftMinimalEvalhubPipeline:
    """Basic tests for SFT Minimal Eval Hub pipeline."""

    def test_pipeline_function_exists(self):
        """Test that the pipeline function is properly defined."""
        assert callable(sft_pipeline_evalhub_easy)

    def test_pipeline_compiles(self, tmp_path):
        """Test that the pipeline compiles successfully."""
        output_path = tmp_path / "pipeline.yaml"
        compiler.Compiler().compile(
            pipeline_func=sft_pipeline_evalhub_easy,
            package_path=str(output_path),
        )
        assert output_path.exists()
