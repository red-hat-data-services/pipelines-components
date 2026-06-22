"""Unit tests for pattern.json and generated notebook artifacts."""

import ast
import json
import re
import sys
from pathlib import Path
from unittest import mock

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


def _notebook_source(notebook_path: Path) -> str:
    """Return concatenated source from all cells in a notebook."""
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    return "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])


def _extract_quoted_assignment(source: str, variable: str) -> str:
    """Extract a string value from ``variable = "..."`` or triple-quoted assignments."""
    patterns = (
        rf'{variable}\s*=\s*"""([^"]*)"""',
        rf"{variable}\s*=\s*'''([^']*)'''",
        rf'{variable}\s*=\s*"([^"]*)"',
        rf"{variable}\s*=\s*'([^']*)'",
    )
    for pattern in patterns:
        match = re.search(pattern, source)
        if match:
            return match.group(1)
    msg = f"Could not find string assignment for {variable!r}"
    raise AssertionError(msg)


def _extract_int_assignment(source: str, variable: str) -> int:
    """Extract an integer value from ``variable = 123``."""
    match = re.search(rf"{variable}\s*=\s*(-?\d+)", source)
    if not match:
        msg = f"Could not find int assignment for {variable!r}"
        raise AssertionError(msg)
    return int(match.group(1))


def _extract_embedding_params(source: str) -> dict:
    """Parse ``OGXEmbeddingParams(**{...})`` from notebook source."""
    match = re.search(r"OGXEmbeddingParams\(\*\*(\{.*?\})\)", source, re.DOTALL)
    if not match:
        msg = "Could not find OGXEmbeddingParams(**{...}) in notebook source"
        raise AssertionError(msg)
    return ast.literal_eval(match.group(1))


def _load_pattern_artifacts(rag_patterns_dir: Path, pattern_name: str) -> dict:
    """Load pattern.json and generated notebooks for one pattern directory."""
    pattern_dir = rag_patterns_dir / pattern_name
    pattern = json.loads((pattern_dir / "pattern.json").read_text(encoding="utf-8"))
    indexing_source = _notebook_source(pattern_dir / "indexing.ipynb")
    inference_source = _notebook_source(pattern_dir / "inference.ipynb")
    return {
        "pattern": pattern,
        "indexing_source": indexing_source,
        "inference_source": inference_source,
    }


def _assert_vector_store_parity(artifacts: dict) -> None:
    """pattern.json vector store binding must match both generated notebooks."""
    binding = artifacts["pattern"]["settings"]["vector_store_binding"]
    provider_id = binding["provider_id"]
    vector_store_id = binding["vector_store_id"]

    for label, source in (
        ("indexing.ipynb", artifacts["indexing_source"]),
        ("inference.ipynb", artifacts["inference_source"]),
    ):
        assert _extract_quoted_assignment(source, "provider_id") == provider_id, label
        assert _extract_quoted_assignment(source, "collection_name") == vector_store_id, label


def _assert_indexing_chunking_and_embedding_parity(artifacts: dict) -> None:
    """Index-building notebook must reflect pattern.json chunking and embedding settings."""
    settings = artifacts["pattern"]["settings"]
    chunking = settings["chunking"]
    embedding = settings["embedding"]
    source = artifacts["indexing_source"]

    assert _extract_quoted_assignment(source, "chunking_method") == chunking["method"]
    assert _extract_int_assignment(source, "chunk_size") == chunking["chunk_size"]
    assert _extract_int_assignment(source, "chunk_overlap") == chunking["chunk_overlap"]
    assert _extract_quoted_assignment(source, "embedding_model_id") == embedding["model_id"]
    assert _extract_quoted_assignment(source, "distance_metric") == embedding["distance_metric"]
    assert _extract_embedding_params(source) == embedding["embedding_params"]


def _assert_inference_retrieval_and_generation_parity(artifacts: dict) -> None:
    """Inference notebook must reflect pattern.json retrieval and generation settings."""
    settings = artifacts["pattern"]["settings"]
    retrieval = settings["retrieval"]
    generation = settings["generation"]
    embedding = settings["embedding"]
    source = artifacts["inference_source"]

    assert _extract_quoted_assignment(source, "embedding_model_id") == embedding["model_id"]
    assert _extract_quoted_assignment(source, "distance_metric") == embedding["distance_metric"]
    assert _extract_embedding_params(source) == embedding["embedding_params"]
    assert _extract_quoted_assignment(source, "method") == retrieval["method"]
    assert _extract_int_assignment(source, "number_of_chunks") == retrieval["number_of_chunks"]
    assert _extract_quoted_assignment(source, "search_mode") == retrieval["search_mode"]
    assert _extract_quoted_assignment(source, "chat_model_id") == generation["model_id"]
    assert _extract_quoted_assignment(source, "system_message_text") == generation["system_message_text"]
    assert _extract_quoted_assignment(source, "user_message_text") == generation["user_message_text"]
    assert _extract_quoted_assignment(source, "context_template_text") == generation["context_template_text"]


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


class TestPatternNotebookParity:
    """Cross-artifact parity between pattern.json and generated notebooks."""

    _ENV = {
        "OGX_CLIENT_BASE_URL": "https://ogx.example.com",
        "OGX_CLIENT_API_KEY": "test-api-key",
    }

    @mock.patch.dict("os.environ", _ENV)
    def test_pattern_json_and_notebooks_share_vector_store_binding(self, tmp_path):
        """vector_store_binding in pattern.json matches provider_id/collection_name in both notebooks."""
        rag_patterns_dir = _run_optimization(
            tmp_path,
            [_make_rich_evaluation("Pattern1", collection="vs-col-parity-001")],
        )
        artifacts = _load_pattern_artifacts(rag_patterns_dir, "Pattern1")
        _assert_vector_store_parity(artifacts)

    @mock.patch.dict("os.environ", _ENV)
    def test_indexing_notebook_matches_pattern_json_chunking_and_embedding(self, tmp_path):
        """indexing.ipynb chunking and embedding values are sourced from pattern.json settings."""
        rag_patterns_dir = _run_optimization(
            tmp_path,
            [_make_rich_evaluation("Pattern1", collection="vs-col-parity-002")],
        )
        artifacts = _load_pattern_artifacts(rag_patterns_dir, "Pattern1")
        _assert_indexing_chunking_and_embedding_parity(artifacts)

    @mock.patch.dict("os.environ", _ENV)
    def test_inference_notebook_matches_pattern_json_retrieval_and_generation(self, tmp_path):
        """inference.ipynb retrieval and generation values are sourced from pattern.json settings."""
        rag_patterns_dir = _run_optimization(
            tmp_path,
            [_make_rich_evaluation("Pattern1", collection="vs-col-parity-003")],
        )
        artifacts = _load_pattern_artifacts(rag_patterns_dir, "Pattern1")
        _assert_inference_retrieval_and_generation_parity(artifacts)

    @mock.patch.dict("os.environ", _ENV)
    def test_indexing_and_inference_notebooks_share_vector_store_from_pattern(self, tmp_path):
        """Both notebooks reuse the same provider_id and collection_name as pattern.json."""
        rag_patterns_dir = _run_optimization(
            tmp_path,
            [_make_rich_evaluation("Pattern1", collection="vs-col-shared-999")],
        )
        artifacts = _load_pattern_artifacts(rag_patterns_dir, "Pattern1")
        binding = artifacts["pattern"]["settings"]["vector_store_binding"]

        indexing_provider = _extract_quoted_assignment(artifacts["indexing_source"], "provider_id")
        indexing_collection = _extract_quoted_assignment(artifacts["indexing_source"], "collection_name")
        inference_provider = _extract_quoted_assignment(artifacts["inference_source"], "provider_id")
        inference_collection = _extract_quoted_assignment(artifacts["inference_source"], "collection_name")

        assert indexing_provider == inference_provider == binding["provider_id"]
        assert indexing_collection == inference_collection == binding["vector_store_id"]

    @mock.patch.dict("os.environ", _ENV)
    def test_notebooks_use_pipeline_input_and_test_data_keys(self, tmp_path):
        """indexing.ipynb and inference.ipynb embed the S3 keys passed to the component."""
        rag_patterns_dir = _run_optimization(
            tmp_path,
            [_make_rich_evaluation("Pattern1", collection="vs-col-keys")],
        )
        artifacts = _load_pattern_artifacts(rag_patterns_dir, "Pattern1")

        assert (
            _extract_quoted_assignment(artifacts["indexing_source"], "input_data_key")
            == "datasets/financial_ibm/documents"
        )
        assert (
            _extract_quoted_assignment(artifacts["inference_source"], "test_data_key")
            == "datasets/financial_ibm/benchmark_data.json"
        )

    @mock.patch.dict("os.environ", _ENV)
    def test_all_artifacts_parity_for_multiple_patterns(self, tmp_path):
        """Each generated pattern directory keeps pattern.json and notebooks in sync."""
        rag_patterns_dir = _run_optimization(
            tmp_path,
            [
                _make_rich_evaluation("PatternA", collection="vs-col-a"),
                _make_rich_evaluation("PatternB", collection="vs-col-b"),
            ],
        )

        for pattern_name, expected_collection in (("PatternA", "vs-col-a"), ("PatternB", "vs-col-b")):
            artifacts = _load_pattern_artifacts(rag_patterns_dir, pattern_name)
            assert artifacts["pattern"]["settings"]["vector_store_binding"]["vector_store_id"] == expected_collection
            _assert_vector_store_parity(artifacts)
            _assert_indexing_chunking_and_embedding_parity(artifacts)
            _assert_inference_retrieval_and_generation_parity(artifacts)
