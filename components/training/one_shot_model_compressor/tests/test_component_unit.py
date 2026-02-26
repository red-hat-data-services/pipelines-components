"""Tests for the open_shot_model_compressor component."""

import sys
from unittest import mock

import pytest

from ..component import one_shot_model_compressor


@pytest.fixture(autouse=True)
def mock_external_modules():
    """Inject mock modules for runtime-only dependencies."""
    modules = {}
    for name in [
        "datasets",
        "compressed_tensors",
        "compressed_tensors.offload",
        "transformers",
        "llmcompressor",
        "llmcompressor.modifiers",
        "llmcompressor.modifiers.quantization",
    ]:
        modules[name] = mock.MagicMock()

    with mock.patch.dict(sys.modules, modules):
        yield modules


class TestOpenShotModelCompressorUnitTests:
    """Unit tests for component logic."""

    def test_component_function_exists(self):
        """Test that the component function is properly imported."""
        assert callable(one_shot_model_compressor)
        assert hasattr(one_shot_model_compressor, "python_func")

    def test_dataset_loaded_with_correct_args(self, mock_external_modules):
        """Test that load_dataset is called with the provided id and split."""
        load_dataset = mock_external_modules["datasets"].load_dataset
        output_model = mock.MagicMock()

        one_shot_model_compressor.python_func(
            model_id="meta-llama/Llama-3-8B",
            recipe="recipe.yaml",
            dataset_id="my-dataset",
            dataset_split="train",
            output_model=output_model,
        )

        load_dataset.assert_called_once_with("my-dataset", split="train")

    def test_dataset_shuffled_with_seed_and_selected(self, mock_external_modules):
        """Test that the dataset is shuffled with the seed and truncated to num_calibration_samples."""
        load_dataset = mock_external_modules["datasets"].load_dataset
        mock_ds = load_dataset.return_value
        output_model = mock.MagicMock()

        one_shot_model_compressor.python_func(
            model_id="meta-llama/Llama-3-8B",
            recipe="recipe.yaml",
            dataset_id="my-dataset",
            dataset_split="train",
            output_model=output_model,
            num_calibration_samples=256,
            seed=123,
        )

        mock_ds.shuffle.assert_called_once_with(seed=123)
        mock_ds.shuffle.return_value.select.assert_called_once_with(range(256))

    def test_model_loaded_from_pretrained(self, mock_external_modules):
        """Test that AutoModelForCausalLM.from_pretrained is called with the model id."""
        auto_model = mock_external_modules["transformers"].AutoModelForCausalLM
        output_model = mock.MagicMock()

        one_shot_model_compressor.python_func(
            model_id="meta-llama/Llama-3-8B",
            recipe="recipe.yaml",
            dataset_id="my-dataset",
            dataset_split="train",
            output_model=output_model,
        )

        auto_model.from_pretrained.assert_called_once_with("meta-llama/Llama-3-8B", dtype="auto")

    def test_tokenizer_loaded_from_pretrained(self, mock_external_modules):
        """Test that AutoTokenizer.from_pretrained is called with the model id."""
        auto_tokenizer = mock_external_modules["transformers"].AutoTokenizer
        output_model = mock.MagicMock()

        one_shot_model_compressor.python_func(
            model_id="meta-llama/Llama-3-8B",
            recipe="recipe.yaml",
            dataset_id="my-dataset",
            dataset_split="train",
            output_model=output_model,
        )

        auto_tokenizer.from_pretrained.assert_called_once_with("meta-llama/Llama-3-8B", trust_remote_code=True)

    def test_oneshot_called_with_correct_args(self, mock_external_modules):
        """Test that oneshot is called with model, dataset, recipe, and tokenizer."""
        oneshot = mock_external_modules["llmcompressor"].oneshot
        auto_model = mock_external_modules["transformers"].AutoModelForCausalLM
        auto_tokenizer = mock_external_modules["transformers"].AutoTokenizer
        output_model = mock.MagicMock()

        one_shot_model_compressor.python_func(
            model_id="meta-llama/Llama-3-8B",
            recipe="recipe.yaml",
            dataset_id="my-dataset",
            dataset_split="train",
            output_model=output_model,
            num_calibration_samples=128,
            max_sequence_length=1024,
        )

        oneshot.assert_called_once()
        call_kwargs = oneshot.call_args[1]
        assert call_kwargs["model"] == auto_model.from_pretrained.return_value
        assert call_kwargs["recipe"] == "recipe.yaml"
        assert call_kwargs["tokenizer"] == auto_tokenizer.from_pretrained.return_value
        assert call_kwargs["max_seq_length"] == 1024
        assert call_kwargs["num_calibration_samples"] == 128

    def test_model_and_tokenizer_saved_to_output_path(self, mock_external_modules):
        """Test that the compressed model and tokenizer are saved to the output artifact path."""
        oneshot = mock_external_modules["llmcompressor"].oneshot
        auto_tokenizer = mock_external_modules["transformers"].AutoTokenizer
        output_model = mock.MagicMock()
        output_model.path = "/tmp/output_model"

        one_shot_model_compressor.python_func(
            model_id="meta-llama/Llama-3-8B",
            recipe="recipe.yaml",
            dataset_id="my-dataset",
            dataset_split="train",
            output_model=output_model,
        )

        oneshot.return_value.save_pretrained.assert_called_once_with("/tmp/output_model")
        auto_tokenizer.from_pretrained.return_value.save_pretrained.assert_called_once_with("/tmp/output_model")

    def test_default_parameter_values(self, mock_external_modules):
        """Test that defaults are applied for num_calibration_samples, max_sequence_length, and seed."""
        load_dataset = mock_external_modules["datasets"].load_dataset
        mock_ds = load_dataset.return_value
        oneshot = mock_external_modules["llmcompressor"].oneshot
        output_model = mock.MagicMock()

        one_shot_model_compressor.python_func(
            model_id="meta-llama/Llama-3-8B",
            recipe="recipe.yaml",
            dataset_id="my-dataset",
            dataset_split="train",
            output_model=output_model,
        )

        # seed default = 42
        mock_ds.shuffle.assert_called_once_with(seed=42)
        # num_calibration_samples default = 512
        mock_ds.shuffle.return_value.select.assert_called_once_with(range(512))
        # max_sequence_length default = 2048, num_calibration_samples default = 512
        call_kwargs = oneshot.call_args[1]
        assert call_kwargs["max_seq_length"] == 2048
        assert call_kwargs["num_calibration_samples"] == 512
