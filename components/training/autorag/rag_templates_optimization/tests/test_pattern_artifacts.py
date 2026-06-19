"""Unit tests for pattern.json and generated notebook artifacts."""

import json
import sys
from unittest import mock

import pytest

from ..component import rag_templates_optimization
from .test_component_unit import _make_all_mocks, _make_ogx_client_module


def _make_rich_evaluation(
    pattern_name: str,
    *,
    final_score: float = 0.91,
    collection: str = "vs-collection-abc",
):
    """Build a mock ai4rag evaluation with indexing and RAG params."""
    evaluation = mock.MagicMock()
    evaluation.pattern_name = pattern_name
    evaluation.indexing_params = {
        "chunking": {"method": "recursive", "chunk_size": 1024, "chunk_overlap": 128},
        "embedding": {
            "model_id": "vllm-embedding/bge-m3",
            "distance_metric": "cosine",
            "embedding_params": {"embedding_dimension": 1024},
        },
    }
    evaluation.rag_params = {
        "retrieval": {"method": "simple", "number_of_chunks": 3, "search_mode": "vector"},
        "generation": {
            "model_id": "vllm-inference/llama",
            "system_message_text": "You are helpful.",
            "user_message_text": "{question}",
            "context_template_text": "{document}",
        },
    }
    evaluation.scores = {"scores": {"faithfulness": {"mean": final_score}}}
    evaluation.final_score = final_score
    evaluation.execution_time = 2.5
    evaluation.collection = collection
    return evaluation


def _run_optimization(tmp_path, evaluations, *, vector_io_provider_id: str = "milvus"):
    """Run rag_templates_optimization with mocked experiment evaluations."""
    mocks = _make_all_mocks()
    ogx_mod = _make_ogx_client_module()
    mock_ogx = mock.MagicMock()
    mock_ogx.models.list.return_value = []
    mock_provider = mock.MagicMock()
    mock_provider.provider_type = "milvus"
    mock_ogx.providers.retrieve.return_value = mock_provider
    ogx_mod.OgxClient.return_value = mock_ogx
    mocks["ogx_client"] = ogx_mod

    mock_exp = mock.MagicMock()
    mock_exp.results.evaluations = evaluations
    mock_exp.results.evaluation_data = [[] for _ in evaluations]
    mock_exp.results.max_combinations = len(evaluations)
    mocks["ai4rag.core.experiment.experiment"].AI4RAGExperiment.return_value = mock_exp

    search_space_report = tmp_path / "report.yml"
    search_space_report.write_text("{}")
    test_data = tmp_path / "test_data.json"
    test_data.write_text("[]")
    extracted_text = tmp_path / "extracted_text"
    extracted_text.mkdir()

    rag_patterns_dir = tmp_path / "rag_patterns"
    rag_patterns_dir.mkdir()
    rag_patterns = mock.MagicMock()
    rag_patterns.path = str(rag_patterns_dir)
    rag_patterns.uri = "s3://bucket/rag_patterns"
    rag_patterns.metadata = {}

    with mock.patch.dict(sys.modules, mocks):
        rag_templates_optimization.python_func(
            extracted_text=str(extracted_text),
            test_data=str(test_data),
            search_space_prep_report=str(search_space_report),
            rag_patterns=rag_patterns,
            test_data_key="datasets/financial_ibm/benchmark_data.json",
            input_data_key="datasets/financial_ibm/documents",
            vector_io_provider_id=vector_io_provider_id,
            optimization_settings={"metric": "faithfulness", "max_number_of_rag_patterns": 8},
        )

    return rag_patterns_dir


class TestPatternArtifacts:
    """Tests for pattern.json content and notebook placeholder substitution."""

    @mock.patch.dict(
        "os.environ",
        {
            "OGX_CLIENT_BASE_URL": "https://ogx.example.com",
            "OGX_CLIENT_API_KEY": "test-api-key",
        },
    )
    def test_pattern_json_includes_vector_store_binding(self, tmp_path):
        """pattern.json stores OGX provider and collection from optimization."""
        rag_patterns_dir = _run_optimization(
            tmp_path,
            [_make_rich_evaluation("Pattern1", collection="vs-col-123")],
        )

        pattern = json.loads((rag_patterns_dir / "Pattern1" / "pattern.json").read_text(encoding="utf-8"))
        binding = pattern["settings"]["vector_store_binding"]
        assert binding["provider_id"] == "milvus"
        assert binding["vector_store_id"] == "vs-col-123"
        assert pattern["settings"]["chunking"]["chunk_size"] == 1024
        assert pattern["settings"]["embedding"]["model_id"] == "vllm-embedding/bge-m3"

    @mock.patch.dict(
        "os.environ",
        {
            "OGX_CLIENT_BASE_URL": "https://ogx.example.com",
            "OGX_CLIENT_API_KEY": "test-api-key",
        },
    )
    def test_generated_indexing_notebook_substitutes_vector_store_and_chunking(self, tmp_path):
        """indexing.ipynb receives provider_id, collection_name, and chunk params from pattern."""
        rag_patterns_dir = _run_optimization(
            tmp_path,
            [_make_rich_evaluation("Pattern1", collection="vs-col-456")],
        )

        notebook_path = rag_patterns_dir / "Pattern1" / "indexing.ipynb"
        assert notebook_path.is_file()
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])

        assert 'provider_id = "milvus"' in source
        assert 'collection_name = "vs-col-456"' in source
        assert "chunk_size = 1024" in source
        assert "chunk_overlap = 128" in source
        assert "{PROVIDER_ID}" not in source
        assert "{COLLECTION_NAME}" not in source
        assert "datasets/financial_ibm/documents" in source

    @mock.patch.dict(
        "os.environ",
        {
            "OGX_CLIENT_BASE_URL": "https://ogx.example.com",
            "OGX_CLIENT_API_KEY": "test-api-key",
        },
    )
    def test_generated_inference_notebook_substitutes_vector_store(self, tmp_path):
        """inference.ipynb reuses the same vector store binding as indexing."""
        rag_patterns_dir = _run_optimization(
            tmp_path,
            [_make_rich_evaluation("Pattern1", collection="vs-col-789")],
        )

        notebook_path = rag_patterns_dir / "Pattern1" / "inference.ipynb"
        assert notebook_path.is_file()
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])

        assert 'provider_id = "milvus"' in source
        assert 'collection_name = "vs-col-789"' in source
        assert "datasets/financial_ibm/benchmark_data.json" in source
