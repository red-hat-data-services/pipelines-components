"""Tests for the test_data_loader thin wrapper component."""

import inspect
import json
from types import SimpleNamespace
from unittest import mock

import pytest

from ..component import test_data_loader

VALID_BENCHMARK_RECORDS = [
    {"question": "What is X?", "correct_answers": ["Answer X"], "correct_answer_document_ids": ["doc_1"]},
    {"question": "What is Y?", "correct_answers": ["Answer Y"], "correct_answer_document_ids": ["doc_2"]},
]


def _make_ai4rag_mocks():
    """Build mock modules for ai4rag.components and ai4rag.components.data."""
    mock_create_s3_client = mock.MagicMock(name="create_s3_client")
    mock_load_test_data = mock.MagicMock(name="load_test_data")

    mock_s3_module = mock.MagicMock()
    mock_s3_module.create_s3_client = mock_create_s3_client

    mock_loader_module = mock.MagicMock()
    mock_loader_module.load_test_data = mock_load_test_data

    modules = {
        "ai4rag": mock.MagicMock(),
        "ai4rag.components": mock.MagicMock(),
        "ai4rag.components.data": mock.MagicMock(),
        "ai4rag.components.data.test_data_loader": mock_loader_module,
        "ai4rag.components.utils": mock.MagicMock(),
        "ai4rag.components.utils.s3": mock_s3_module,
    }
    return modules, mock_create_s3_client, mock_load_test_data


class TestTestDataLoaderUnitTests:
    """Unit tests for the test_data_loader thin wrapper."""

    def test_component_function_exists(self):
        """Component factory exists and exposes python_func."""
        assert callable(test_data_loader)
        assert hasattr(test_data_loader, "python_func")

    def test_component_with_default_parameters(self):
        """Component has expected required parameters in interface."""
        sig = inspect.signature(test_data_loader.python_func)
        params = list(sig.parameters)
        assert "test_data_bucket_name" in params
        assert "test_data_path" in params
        assert "benchmark_sample_size" in params
        assert sig.parameters["benchmark_sample_size"].default == 25

    def test_delegates_to_ai4rag_load_test_data(self, tmp_path):
        """Wrapper calls create_s3_client and load_test_data with correct args."""
        modules, mock_create_s3, mock_load = _make_ai4rag_mocks()
        mock_s3_client = mock.MagicMock(name="s3_client_instance")
        mock_create_s3.return_value = mock_s3_client
        mock_load.return_value = SimpleNamespace(data=VALID_BENCHMARK_RECORDS)

        out_path = tmp_path / "test_data.json"
        artifact = mock.MagicMock()
        artifact.path = str(out_path)

        with mock.patch.dict("sys.modules", modules):
            test_data_loader.python_func(
                test_data_bucket_name="my-bucket",
                test_data_path="data/test.json",
                benchmark_sample_size=10,
                test_data=artifact,
            )

        mock_create_s3.assert_called_once()
        mock_load.assert_called_once_with(
            bucket_name="my-bucket",
            key="data/test.json",
            benchmark_sample_size=10,
            s3_client=mock_s3_client,
        )

    def test_writes_result_to_artifact_path(self, tmp_path):
        """Output JSON is written to the artifact path."""
        modules, mock_create_s3, mock_load = _make_ai4rag_mocks()
        mock_create_s3.return_value = mock.MagicMock()
        mock_load.return_value = SimpleNamespace(data=VALID_BENCHMARK_RECORDS)

        out_path = tmp_path / "test_data.json"
        artifact = mock.MagicMock()
        artifact.path = str(out_path)

        with mock.patch.dict("sys.modules", modules):
            test_data_loader.python_func(
                test_data_bucket_name="my-bucket",
                test_data_path="data/test.json",
                test_data=artifact,
            )

        assert out_path.exists()
        result = json.loads(out_path.read_text(encoding="utf-8"))
        assert result == VALID_BENCHMARK_RECORDS

    def test_default_benchmark_sample_size(self, tmp_path):
        """Default benchmark_sample_size=25 is passed to load_test_data."""
        modules, mock_create_s3, mock_load = _make_ai4rag_mocks()
        mock_create_s3.return_value = mock.MagicMock()
        mock_load.return_value = SimpleNamespace(data=[])

        artifact = mock.MagicMock()
        artifact.path = str(tmp_path / "test_data.json")

        with mock.patch.dict("sys.modules", modules):
            test_data_loader.python_func(
                test_data_bucket_name="bucket",
                test_data_path="key.json",
                test_data=artifact,
            )

        assert mock_load.call_args.kwargs["benchmark_sample_size"] == 25

    def test_propagates_ai4rag_exception(self, tmp_path):
        """Exceptions from ai4rag are propagated to the caller."""
        modules, mock_create_s3, mock_load = _make_ai4rag_mocks()
        mock_create_s3.return_value = mock.MagicMock()
        mock_load.side_effect = FileNotFoundError("Test data object not found in S3")

        artifact = mock.MagicMock()
        artifact.path = str(tmp_path / "test_data.json")

        with mock.patch.dict("sys.modules", modules):
            with pytest.raises(FileNotFoundError, match="Test data object not found"):
                test_data_loader.python_func(
                    test_data_bucket_name="my-bucket",
                    test_data_path="missing/test.json",
                    test_data=artifact,
                )
