"""Tests for the documents_discovery thin wrapper component."""

import inspect
import json
from unittest import mock

import pytest

from ..component import documents_discovery


def _make_ai4rag_mocks():
    """Build mock modules for ai4rag.components and ai4rag.components.data."""
    mock_create_s3_client = mock.MagicMock(name="create_s3_client")
    mock_discover_documents = mock.MagicMock(name="discover_documents")

    mock_s3_module = mock.MagicMock()
    mock_s3_module.create_s3_client = mock_create_s3_client

    mock_discovery_module = mock.MagicMock()
    mock_discovery_module.discover_documents = mock_discover_documents

    modules = {
        "ai4rag": mock.MagicMock(),
        "ai4rag.components": mock.MagicMock(),
        "ai4rag.components.data": mock.MagicMock(),
        "ai4rag.components.data.documents_discovery": mock_discovery_module,
        "ai4rag.components.utils": mock.MagicMock(),
        "ai4rag.components.utils.s3": mock_s3_module,
    }
    return modules, mock_create_s3_client, mock_discover_documents


class TestDocumentsDiscoveryUnitTests:
    """Unit tests for the documents_discovery thin wrapper."""

    def test_component_function_exists(self):
        """Component factory exists and exposes python_func."""
        assert callable(documents_discovery)
        assert hasattr(documents_discovery, "python_func")

    def test_component_with_default_parameters(self):
        """Component has expected required interface."""
        sig = inspect.signature(documents_discovery.python_func)
        params = list(sig.parameters)
        assert "input_data_bucket_name" in params
        assert "input_data_path" in params

    def test_delegates_to_ai4rag_discover_documents(self, tmp_path):
        """Wrapper calls create_s3_client and discover_documents with correct args."""
        modules, mock_create_s3, mock_discover = _make_ai4rag_mocks()
        mock_s3_client = mock.MagicMock(name="s3_client_instance")
        mock_create_s3.return_value = mock_s3_client
        mock_result = mock.MagicMock()
        mock_discover.return_value = mock_result

        discovered = mock.MagicMock()
        discovered.path = str(tmp_path / "descriptor")

        with mock.patch.dict("sys.modules", modules):
            documents_discovery.python_func(
                input_data_bucket_name="my-bucket",
                input_data_path="docs/",
                sampling_enabled=True,
                sampling_max_size=2.5,
                discovered_documents=discovered,
            )

        mock_create_s3.assert_called_once()
        mock_discover.assert_called_once_with(
            bucket_name="my-bucket",
            prefix="docs/",
            test_data_doc_names=None,
            sampling_enabled=True,
            sampling_max_size_gb=2.5,
            s3_client=mock_s3_client,
        )

    def test_saves_result_to_artifact_path(self, tmp_path):
        """DiscoveryResult.save is called with the correct output path."""
        modules, mock_create_s3, mock_discover = _make_ai4rag_mocks()
        mock_create_s3.return_value = mock.MagicMock()
        mock_result = mock.MagicMock()
        mock_discover.return_value = mock_result

        discovered = mock.MagicMock()
        discovered.path = str(tmp_path / "descriptor")

        with mock.patch.dict("sys.modules", modules):
            documents_discovery.python_func(
                input_data_bucket_name="my-bucket",
                input_data_path="docs/",
                discovered_documents=discovered,
            )

        expected_dir = tmp_path / "descriptor"
        assert expected_dir.exists()
        mock_result.save.assert_called_once_with(
            path=expected_dir,
            filename="documents_descriptor.json",
        )

    def test_extracts_test_data_doc_names(self, tmp_path):
        """Test data doc names are extracted and passed to discover_documents."""
        modules, mock_create_s3, mock_discover = _make_ai4rag_mocks()
        mock_create_s3.return_value = mock.MagicMock()
        mock_discover.return_value = mock.MagicMock()

        test_data_json = [
            {"question": "q1", "correct_answer_document_ids": ["doc_a.pdf", "doc_b.pdf"]},
            {"question": "q2", "correct_answer_document_ids": ["doc_a.pdf", "doc_c.txt"]},
        ]
        td_file = tmp_path / "test_data.json"
        td_file.write_text(json.dumps(test_data_json), encoding="utf-8")
        test_data_artifact = mock.MagicMock()
        test_data_artifact.path = str(td_file)

        discovered = mock.MagicMock()
        discovered.path = str(tmp_path / "descriptor")

        with mock.patch.dict("sys.modules", modules):
            documents_discovery.python_func(
                input_data_bucket_name="my-bucket",
                input_data_path="docs/",
                test_data=test_data_artifact,
                discovered_documents=discovered,
            )

        call_kwargs = mock_discover.call_args.kwargs
        passed_names = set(call_kwargs["test_data_doc_names"])
        assert passed_names == {"doc_a.pdf", "doc_b.pdf", "doc_c.txt"}

    def test_no_test_data_passes_none(self, tmp_path):
        """When test_data is None, test_data_doc_names=None is passed."""
        modules, mock_create_s3, mock_discover = _make_ai4rag_mocks()
        mock_create_s3.return_value = mock.MagicMock()
        mock_discover.return_value = mock.MagicMock()

        discovered = mock.MagicMock()
        discovered.path = str(tmp_path / "descriptor")

        with mock.patch.dict("sys.modules", modules):
            documents_discovery.python_func(
                input_data_bucket_name="my-bucket",
                input_data_path="docs/",
                test_data=None,
                discovered_documents=discovered,
            )

        assert mock_discover.call_args.kwargs["test_data_doc_names"] is None

    def test_propagates_ai4rag_exception(self, tmp_path):
        """Exceptions from ai4rag are propagated to the caller."""
        modules, mock_create_s3, mock_discover = _make_ai4rag_mocks()
        mock_create_s3.return_value = mock.MagicMock()
        mock_discover.side_effect = ValueError("No documents to process")

        discovered = mock.MagicMock()
        discovered.path = str(tmp_path / "descriptor")

        with mock.patch.dict("sys.modules", modules):
            with pytest.raises(ValueError, match="No documents to process"):
                documents_discovery.python_func(
                    input_data_bucket_name="my-bucket",
                    input_data_path="docs/",
                    discovered_documents=discovered,
                )
