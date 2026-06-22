"""Tests for the search_space_preparation component."""

import ssl
import sys
import types
from unittest import mock

import pytest

from ..component import search_space_preparation


class _SentinelAbort(Exception):
    """Raised by mocks to abort the component after client creation."""


def _make_httpx_module():
    """Return a minimal fake httpx module with a trackable Client class."""
    mod = types.ModuleType("httpx")

    class ConnectError(Exception):
        pass

    class Client:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    mod.ConnectError = ConnectError
    mod.Client = Client
    return mod


def _make_minimal_httpx_module():
    """Return a minimal httpx stub for validation-only test paths."""
    mod = types.ModuleType("httpx")

    class ConnectError(Exception):
        pass

    class Client:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    mod.ConnectError = ConnectError
    mod.Client = Client
    return mod


def _make_ogx_client_module():
    """Stub ogx_client with a real APIConnectionError (MagicMock breaks except clauses)."""
    mod = types.ModuleType("ogx_client")

    class APIConnectionError(Exception):
        pass

    mod.APIConnectionError = APIConnectionError
    mod.OgxClient = mock.MagicMock()
    return mod


def _make_all_mocks():
    """Build sys.modules patch dict for all heavy dependencies."""
    mocks = {}
    for name in [
        "pysqlite3",
        "ai4rag",
        "ai4rag.core",
        "ai4rag.core.experiment",
        "ai4rag.core.experiment.benchmark_data",
        "ai4rag.core.experiment.mps",
        "ai4rag.rag",
        "ai4rag.rag.embedding",
        "ai4rag.rag.embedding.base_model",
        "ai4rag.rag.embedding.ogx",
        "ai4rag.rag.foundation_models",
        "ai4rag.rag.foundation_models.base_model",
        "ai4rag.rag.foundation_models.ogx",
        "ai4rag.search_space",
        "ai4rag.search_space.prepare",
        "ai4rag.search_space.prepare.prepare_search_space",
        "ai4rag.search_space.src",
        "ai4rag.search_space.src.parameter",
        "ai4rag.search_space.src.search_space",
        "langchain_core",
        "langchain_core.documents",
        "pandas",
        "yaml",
    ]:
        mocks[name] = mock.MagicMock()

    httpx_mod = _make_httpx_module()
    mocks["httpx"] = httpx_mod
    return mocks


def _minimal_dependency_modules():
    """Mock imported heavy third-party modules for validation-path tests."""
    return {
        "pandas": mock.MagicMock(),
        "yaml": mock.MagicMock(),
        "ai4rag": mock.MagicMock(),
        "ai4rag.core": mock.MagicMock(),
        "ai4rag.core.experiment": mock.MagicMock(),
        "ai4rag.core.experiment.benchmark_data": mock.MagicMock(BenchmarkData=mock.MagicMock()),
        "ai4rag.core.experiment.mps": mock.MagicMock(ModelsPreSelector=mock.MagicMock()),
        "ai4rag.rag": mock.MagicMock(),
        "ai4rag.rag.embedding": mock.MagicMock(),
        "ai4rag.rag.embedding.base_model": mock.MagicMock(BaseEmbeddingModel=mock.MagicMock()),
        "ai4rag.rag.embedding.ogx": mock.MagicMock(OGXEmbeddingModel=mock.MagicMock()),
        "ai4rag.rag.foundation_models": mock.MagicMock(),
        "ai4rag.rag.foundation_models.base_model": mock.MagicMock(BaseFoundationModel=mock.MagicMock()),
        "ai4rag.rag.foundation_models.ogx": mock.MagicMock(OGXFoundationModel=mock.MagicMock()),
        "ai4rag.search_space": mock.MagicMock(),
        "ai4rag.search_space.prepare": mock.MagicMock(),
        "ai4rag.search_space.prepare.prepare_search_space": mock.MagicMock(
            prepare_search_space_with_ogx=mock.MagicMock()
        ),
        "ai4rag.search_space.src": mock.MagicMock(),
        "ai4rag.search_space.src.parameter": mock.MagicMock(Parameter=mock.MagicMock()),
        "ai4rag.search_space.src.search_space": mock.MagicMock(AI4RAGSearchSpace=mock.MagicMock()),
        "langchain_core": mock.MagicMock(),
        "langchain_core.documents": mock.MagicMock(Document=mock.MagicMock()),
        "ogx_client": mock.MagicMock(OgxClient=mock.MagicMock()),
        "httpx": _make_minimal_httpx_module(),
    }


class TestSearchSpacePreparationUnitTests:
    """Unit tests for component logic."""

    def test_component_function_exists(self):
        """Test that the component function is properly imported."""
        assert callable(search_space_preparation)
        assert hasattr(search_space_preparation, "python_func")

    def test_component_with_default_parameters(self):
        """Test component has expected interface (required args)."""
        import inspect

        sig = inspect.signature(search_space_preparation.python_func)
        params = list(sig.parameters)
        assert "test_data" in params
        assert "extracted_text" in params
        assert "search_space_prep_report" in params

    def test_non_list_embedding_models_raises_type_error(self):
        """embedding_models must be a list when provided."""
        with mock.patch.dict(sys.modules, _minimal_dependency_modules()):
            with pytest.raises(TypeError, match="embedding_models must be a list"):
                search_space_preparation.python_func(
                    test_data=mock.MagicMock(path="/tmp/test_data.json"),
                    extracted_text=mock.MagicMock(path="/tmp/extracted"),
                    search_space_prep_report=mock.MagicMock(path="/tmp/report.yaml"),
                    embedding_models="not-a-list",
                )

    def test_non_list_generation_models_raises_type_error(self):
        """generation_models must be a list."""
        with mock.patch.dict(sys.modules, _minimal_dependency_modules()):
            with pytest.raises(TypeError, match="generation_models must be a list"):
                search_space_preparation.python_func(
                    test_data=mock.MagicMock(path="/tmp/test_data.json"),
                    extracted_text=mock.MagicMock(path="/tmp/extracted"),
                    search_space_prep_report=mock.MagicMock(path="/tmp/report.yaml"),
                    generation_models="not-a-list",
                )

    def test_unsupported_metric_raises_value_error(self):
        """Unsupported metric value raises ValueError with supported list."""
        with mock.patch.dict(sys.modules, _minimal_dependency_modules()):
            with pytest.raises(ValueError, match="is not supported"):
                search_space_preparation.python_func(
                    test_data=mock.MagicMock(path="/tmp/test_data.json"),
                    extracted_text=mock.MagicMock(path="/tmp/extracted"),
                    search_space_prep_report=mock.MagicMock(path="/tmp/report.yaml"),
                    embedding_models=["embed-a"],
                    generation_models=["gen-a"],
                    metric="unsupported_metric",
                )

    @mock.patch.dict(
        "os.environ",
        {
            "OGX_CLIENT_BASE_URL": "https://ogx.example.com",
            "OGX_CLIENT_API_KEY": "test-api-key",
        },
    )
    def test_no_models_with_ogx_does_not_raise(self, tmp_path):
        """When no model lists are provided, OGX auto-discovers models — no early validation error."""
        mocks = _make_all_mocks()

        ogx_mod = _make_ogx_client_module()
        mock_ogx = mock.MagicMock()
        mock_ogx.models.list.return_value = []
        ogx_mod.OgxClient.return_value = mock_ogx
        mocks["ogx_client"] = ogx_mod

        # Abort after search space preparation to avoid full pipeline execution
        mocks[
            "ai4rag.search_space.prepare.prepare_search_space"
        ].prepare_search_space_with_ogx.side_effect = _SentinelAbort

        with mock.patch.dict(sys.modules, mocks):
            with pytest.raises(_SentinelAbort):
                search_space_preparation.python_func(
                    test_data=mock.MagicMock(path=str(tmp_path / "test_data.json")),
                    extracted_text=mock.MagicMock(path=str(tmp_path / "extracted")),
                    search_space_prep_report=mock.MagicMock(path=str(tmp_path / "report.yml")),
                )

    @mock.patch.dict(
        "os.environ",
        {
            "OGX_CLIENT_BASE_URL": "https://ogx.example.com",
            "OGX_CLIENT_API_KEY": "test-api-key",
        },
    )
    def test_partial_model_lists_with_ogx(self, tmp_path):
        """Only generation_models provided — OGX discovers embedding models automatically."""
        mocks = _make_all_mocks()

        ogx_mod = _make_ogx_client_module()
        mock_ogx = mock.MagicMock()
        mock_ogx.models.list.return_value = []
        ogx_mod.OgxClient.return_value = mock_ogx
        mocks["ogx_client"] = ogx_mod

        mocks[
            "ai4rag.search_space.prepare.prepare_search_space"
        ].prepare_search_space_with_ogx.side_effect = _SentinelAbort

        with mock.patch.dict(sys.modules, mocks):
            with pytest.raises(_SentinelAbort):
                search_space_preparation.python_func(
                    test_data=mock.MagicMock(path=str(tmp_path / "test_data.json")),
                    extracted_text=mock.MagicMock(path=str(tmp_path / "extracted")),
                    search_space_prep_report=mock.MagicMock(path=str(tmp_path / "report.yml")),
                    generation_models=["gen-model-a"],
                )

    def test_missing_ogx_env_vars_raises_value_error(self):
        """Missing OGX environment variables raises ValueError."""
        with mock.patch.dict(sys.modules, _minimal_dependency_modules()):
            with mock.patch.dict("os.environ", {}, clear=True):
                with pytest.raises(ValueError, match="OGX_CLIENT_BASE_URL and OGX_CLIENT_API_KEY"):
                    search_space_preparation.python_func(
                        test_data=mock.MagicMock(path="/tmp/test_data.json"),
                        extracted_text=mock.MagicMock(path="/tmp/extracted"),
                        search_space_prep_report=mock.MagicMock(path="/tmp/report.yaml"),
                    )


class TestLanguageDetection:
    """Tests for LLM-based language detection in search_space_preparation.

    The component runs to completion (writing the YAML report) so we can read
    back detected_language from the output and assert its value.
    """

    def _make_ogx_with_chat(self, llm_response_text):
        """Build OGX mock that returns a chat completion with given text."""
        ogx_mod = _make_ogx_client_module()

        mock_model = mock.MagicMock()
        mock_model.identifier = "test-llm"
        mock_model.model_type = "llm"

        mock_models_response = mock.MagicMock()
        mock_models_response.data = [mock_model]

        mock_choice = mock.MagicMock()
        mock_choice.message.content = llm_response_text
        mock_chat_response = mock.MagicMock()
        mock_chat_response.choices = [mock_choice]

        mock_client = mock.MagicMock()
        mock_client.models.list.return_value = mock_models_response
        mock_client.chat.completions.create.return_value = mock_chat_response
        ogx_mod.OgxClient.return_value = mock_client

        return ogx_mod, mock_client

    def _setup_and_run(self, tmp_path, llm_response_text, ogx_mod=None, mock_client=None, **run_kwargs):
        """Run the component to completion and return (detected_language, mock_client).

        Lets prepare_search_space_with_ogx return a minimal mock search space,
        then the component writes the YAML report. We read it back to get detected_language.
        """
        import pandas as pd

        mocks = _make_all_mocks()

        benchmark_df = pd.DataFrame(
            [{"question": "Was ist das?", "correct_answers": [["etwas"]], "correct_answer_document_ids": [["d"]]}]
        )
        mock_pd = mock.MagicMock()
        mock_pd.read_json.return_value = benchmark_df
        mock_pd.DataFrame = pd.DataFrame
        mocks["pandas"] = mock_pd

        if ogx_mod is None:
            ogx_mod, mock_client = self._make_ogx_with_chat(llm_response_text)
        mocks["ogx_client"] = ogx_mod

        mock_search_space = mock.MagicMock()
        mock_search_space.__getitem__ = mock.MagicMock(return_value=mock.MagicMock(values=[]))
        mock_search_space._search_space = {}
        mocks[
            "ai4rag.search_space.prepare.prepare_search_space"
        ].prepare_search_space_with_ogx.return_value = mock_search_space

        mock_yaml = mock.MagicMock()
        captured = {}

        def fake_safe_dump(data, stream):
            captured.update(data)

        mock_yaml.safe_dump = fake_safe_dump
        mock_yaml.safe_load = mock.MagicMock(return_value={})
        mocks["yaml"] = mock_yaml

        test_data_art = mock.MagicMock(path=str(tmp_path / "test_data.json"))
        extracted_art = mock.MagicMock(path=str(tmp_path / "ext"))
        report_path = tmp_path / "report.yml"
        report_art = mock.MagicMock(path=str(report_path))

        with (
            mock.patch.dict(sys.modules, mocks),
            mock.patch.dict(
                "os.environ", {"OGX_CLIENT_BASE_URL": "https://ogx.example.com", "OGX_CLIENT_API_KEY": "key"}
            ),
        ):
            search_space_preparation.python_func(
                test_data=test_data_art,
                extracted_text=extracted_art,
                search_space_prep_report=report_art,
                **run_kwargs,
            )

        return captured.get("detected_language"), mock_client

    def test_detects_german(self, tmp_path):
        """LLM returns 'de' → detected_language is {code: de, name: German}."""
        detected, mock_client = self._setup_and_run(tmp_path, "de")
        assert detected == {"code": "de", "name": "German"}
        mock_client.chat.completions.create.assert_called_once()

    def test_english_returns_english_dict(self, tmp_path):
        """LLM returns 'en' → detected_language is {code: en, name: English}."""
        detected, _ = self._setup_and_run(tmp_path, "en")
        assert detected == {"code": "en", "name": "English"}

    def test_unsupported_code_returns_none(self, tmp_path):
        """LLM returns unsupported code 'fi' → detected_language is None (not written)."""
        detected, _ = self._setup_and_run(tmp_path, "fi")
        assert detected is None

    def test_llm_failure_returns_none(self, tmp_path):
        """When LLM call fails, detected_language is None."""
        ogx_mod = _make_ogx_client_module()
        mock_model = mock.MagicMock()
        mock_model.identifier = "test-llm"
        mock_model.model_type = "llm"
        mock_models_response = mock.MagicMock()
        mock_models_response.data = [mock_model]
        mock_client = mock.MagicMock()
        mock_client.models.list.return_value = mock_models_response
        mock_client.chat.completions.create.side_effect = RuntimeError("LLM unavailable")
        ogx_mod.OgxClient.return_value = mock_client

        detected, _ = self._setup_and_run(tmp_path, "", ogx_mod=ogx_mod, mock_client=mock_client)
        assert detected is None

    def test_prefers_allowed_generation_models(self, tmp_path):
        """When generation_models is provided, uses that model for detection."""
        ogx_mod = _make_ogx_client_module()
        mock_embed = mock.MagicMock()
        mock_embed.identifier = "embed-model"
        mock_embed.model_type = "embedding"
        mock_llm = mock.MagicMock()
        mock_llm.identifier = "preferred-llm"
        mock_llm.model_type = "llm"
        mock_models_response = mock.MagicMock()
        mock_models_response.data = [mock_embed, mock_llm]
        mock_choice = mock.MagicMock()
        mock_choice.message.content = "de"
        mock_chat_response = mock.MagicMock()
        mock_chat_response.choices = [mock_choice]
        mock_client = mock.MagicMock()
        mock_client.models.list.return_value = mock_models_response
        mock_client.chat.completions.create.return_value = mock_chat_response
        ogx_mod.OgxClient.return_value = mock_client

        detected, _ = self._setup_and_run(
            tmp_path, "de", ogx_mod=ogx_mod, mock_client=mock_client, generation_models=["preferred-llm"]
        )
        assert detected == {"code": "de", "name": "German"}
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["model"] == "preferred-llm"


class TestSSLFallbackSearchSpacePreparation:
    """Tests for SSL retry logic in _create_ogx_client."""

    def _make_artifacts(self, tmp_path):
        test_data = mock.MagicMock()
        test_data.path = str(tmp_path / "test_data.json")
        extracted_text = mock.MagicMock()
        extracted_text.path = str(tmp_path / "extracted")
        search_space_prep_report = mock.MagicMock()
        search_space_prep_report.path = str(tmp_path / "report.yml")
        return test_data, extracted_text, search_space_prep_report

    @mock.patch.dict(
        "os.environ",
        {
            "OGX_CLIENT_BASE_URL": "https://ogx.example.com",
            "OGX_CLIENT_API_KEY": "test-api-key",
        },
    )
    def test_ogx_client_ssl_retry_with_verify_false(self, tmp_path):
        """SSL error on models.list() retries OgxClient with verify=False."""
        mocks = _make_all_mocks()

        mock_ogx_client_ok = mock.MagicMock()
        mock_ogx_client_fail = mock.MagicMock()
        mock_ogx_client_fail.models.list.side_effect = ssl.SSLCertVerificationError(
            "CERTIFICATE_VERIFY_FAILED: self-signed certificate"
        )
        mock_ogx_client_ok.models.list.return_value = []

        ogx_call_count = 0
        ogx_kwargs_history = []

        def fake_ogx_client(**kwargs):
            nonlocal ogx_call_count
            ogx_call_count += 1
            ogx_kwargs_history.append(kwargs)
            if ogx_call_count == 1:
                return mock_ogx_client_fail
            return mock_ogx_client_ok

        ogx_mod = _make_ogx_client_module()
        ogx_mod.OgxClient.side_effect = fake_ogx_client

        # Abort after client creation by making prepare_search_space_with_ogx raise
        mocks[
            "ai4rag.search_space.prepare.prepare_search_space"
        ].prepare_search_space_with_ogx.side_effect = _SentinelAbort
        mocks["ogx_client"] = ogx_mod

        test_data, extracted_text, search_space_prep_report = self._make_artifacts(tmp_path)

        with mock.patch.dict(sys.modules, mocks):
            with pytest.raises(_SentinelAbort):
                search_space_preparation.python_func(
                    test_data=test_data,
                    extracted_text=extracted_text,
                    search_space_prep_report=search_space_prep_report,
                )

        assert ogx_call_count == 2, "OgxClient should be instantiated twice (initial + SSL retry)"
        assert ogx_kwargs_history[0].get("http_client") is None, "First call should not disable SSL"
        assert isinstance(ogx_kwargs_history[1].get("http_client"), mocks["httpx"].Client), (
            "Second call should pass httpx.Client"
        )
        assert ogx_kwargs_history[1]["http_client"].kwargs.get("verify") is False

    @mock.patch.dict(
        "os.environ",
        {
            "OGX_CLIENT_BASE_URL": "https://ogx.example.com",
            "OGX_CLIENT_API_KEY": "test-api-key",
        },
    )
    def test_ogx_client_api_connection_error_wrapping_ssl_retries(self, tmp_path):
        """OGXAPIConnectionError wrapping an SSL cause triggers the verify=False retry (production case)."""
        mocks = _make_all_mocks()

        ogx_mod = _make_ogx_client_module()
        OGXAPIConnectionError = ogx_mod.APIConnectionError
        ssl_err = ssl.SSLCertVerificationError("CERTIFICATE_VERIFY_FAILED: self-signed certificate")
        api_err = OGXAPIConnectionError("Connection error.")
        api_err.__cause__ = ssl_err

        mock_ogx_client_fail = mock.MagicMock()
        mock_ogx_client_fail.models.list.side_effect = api_err
        mock_ogx_client_ok = mock.MagicMock()
        mock_ogx_client_ok.models.list.return_value = []

        ogx_call_count = 0
        ogx_kwargs_history = []

        def fake_ogx_client(**kwargs):
            nonlocal ogx_call_count
            ogx_call_count += 1
            ogx_kwargs_history.append(kwargs)
            if ogx_call_count == 1:
                return mock_ogx_client_fail
            return mock_ogx_client_ok

        ogx_mod.OgxClient.side_effect = fake_ogx_client
        mocks["ogx_client"] = ogx_mod

        mocks[
            "ai4rag.search_space.prepare.prepare_search_space"
        ].prepare_search_space_with_ogx.side_effect = _SentinelAbort

        test_data, extracted_text, search_space_prep_report = self._make_artifacts(tmp_path)

        with mock.patch.dict(sys.modules, mocks):
            with pytest.raises(_SentinelAbort):
                search_space_preparation.python_func(
                    test_data=test_data,
                    extracted_text=extracted_text,
                    search_space_prep_report=search_space_prep_report,
                )

        assert ogx_call_count == 2, "OgxClient should be instantiated twice (initial + SSL retry)"
        assert ogx_kwargs_history[0].get("http_client") is None
        assert isinstance(ogx_kwargs_history[1].get("http_client"), mocks["httpx"].Client)
        assert ogx_kwargs_history[1]["http_client"].kwargs.get("verify") is False

    @mock.patch.dict(
        "os.environ",
        {
            "OGX_CLIENT_BASE_URL": "https://ogx.example.com",
            "OGX_CLIENT_API_KEY": "test-api-key",
        },
    )
    def test_ogx_client_non_ssl_error_is_reraised(self, tmp_path):
        """Non-SSL error from models.list() is not swallowed — it propagates."""
        mocks = _make_all_mocks()

        mock_ogx_client = mock.MagicMock()
        mock_ogx_client.models.list.side_effect = ConnectionRefusedError("Connection refused")

        ogx_mod = _make_ogx_client_module()
        ogx_mod.OgxClient.return_value = mock_ogx_client
        mocks["ogx_client"] = ogx_mod

        test_data, extracted_text, search_space_prep_report = self._make_artifacts(tmp_path)

        with mock.patch.dict(sys.modules, mocks):
            with pytest.raises(ConnectionRefusedError):
                search_space_preparation.python_func(
                    test_data=test_data,
                    extracted_text=extracted_text,
                    search_space_prep_report=search_space_prep_report,
                )

    @mock.patch.dict(
        "os.environ",
        {
            "OGX_CLIENT_BASE_URL": "https://ogx.example.com",
            "OGX_CLIENT_API_KEY": "test-api-key",
        },
    )
    def test_ogx_client_api_connection_error_non_ssl_cause_is_reraised(self, tmp_path):
        """OGXAPIConnectionError whose cause is NOT SSL propagates without retry."""
        mocks = _make_all_mocks()

        ogx_mod = _make_ogx_client_module()
        OGXAPIConnectionError = ogx_mod.APIConnectionError
        err = OGXAPIConnectionError("Connection timeout")
        err.__cause__ = TimeoutError("timed out")

        mock_ogx_client = mock.MagicMock()
        mock_ogx_client.models.list.side_effect = err
        ogx_mod.OgxClient.return_value = mock_ogx_client
        mocks["ogx_client"] = ogx_mod

        test_data, extracted_text, search_space_prep_report = self._make_artifacts(tmp_path)

        with mock.patch.dict(sys.modules, mocks):
            with pytest.raises(OGXAPIConnectionError):
                search_space_preparation.python_func(
                    test_data=test_data,
                    extracted_text=extracted_text,
                    search_space_prep_report=search_space_prep_report,
                )


class TestSearchSpaceReport:
    """Tests for YAML report content and ModelsPreSelector limiting."""

    @staticmethod
    def _make_ogx_with_chat(response_text: str = "en"):
        """Return ogx module stub whose chat completion returns *response_text*."""
        ogx_mod = _make_ogx_client_module()
        mock_model = mock.MagicMock()
        mock_model.identifier = "test-llm"
        mock_model.model_type = "llm"
        mock_models_response = mock.MagicMock()
        mock_models_response.data = [mock_model]
        mock_choice = mock.MagicMock()
        mock_choice.message.content = response_text
        mock_chat_response = mock.MagicMock()
        mock_chat_response.choices = [mock_choice]
        mock_client = mock.MagicMock()
        mock_client.models.list.return_value = mock_models_response
        mock_client.chat.completions.create.return_value = mock_chat_response
        ogx_mod.OgxClient.return_value = mock_client
        return ogx_mod

    def _run_with_search_space(self, tmp_path, mock_search_space, **run_kwargs):
        """Run component to completion and return captured YAML report dict."""
        import pandas as pd

        mocks = _make_all_mocks()

        benchmark_df = pd.DataFrame(
            [
                {
                    "question": "What is X?",
                    "correct_answers": [["Answer"]],
                    "correct_answer_document_ids": [["doc1"]],
                }
            ]
        )
        mock_pd = mock.MagicMock()
        mock_pd.read_json.return_value = benchmark_df
        mock_pd.DataFrame = pd.DataFrame
        mocks["pandas"] = mock_pd

        mocks["ogx_client"] = self._make_ogx_with_chat()

        mocks[
            "ai4rag.search_space.prepare.prepare_search_space"
        ].prepare_search_space_with_ogx.return_value = mock_search_space

        captured = {}

        def fake_safe_dump(data, stream):
            captured.update(data)

        mock_yaml = mock.MagicMock()
        mock_yaml.safe_dump = fake_safe_dump
        mocks["yaml"] = mock_yaml

        extracted_dir = tmp_path / "extracted"
        extracted_dir.mkdir()
        (extracted_dir / "doc0.md").write_text("sample text", encoding="utf-8")

        test_data_art = mock.MagicMock(path=str(tmp_path / "test_data.json"))
        extracted_art = mock.MagicMock(path=str(extracted_dir))
        report_art = mock.MagicMock(path=str(tmp_path / "report.yml"))

        with (
            mock.patch.dict(sys.modules, mocks),
            mock.patch.dict(
                "os.environ",
                {"OGX_CLIENT_BASE_URL": "https://ogx.example.com", "OGX_CLIENT_API_KEY": "test-api-key"},
            ),
        ):
            search_space_preparation.python_func(
                test_data=test_data_art,
                extracted_text=extracted_art,
                search_space_prep_report=report_art,
                **run_kwargs,
            )

        return captured, mocks

    @staticmethod
    def _make_search_space(*, n_generation: int = 1, n_embedding: int = 1):
        """Build a mock AI4RAGSearchSpace with chunking/retrieval params."""
        mock_foundation = mock.MagicMock()
        mock_foundation.values = [f"gen-{i}" for i in range(n_generation)]
        mock_embedding = mock.MagicMock()
        mock_embedding.values = [f"emb-{i}" for i in range(n_embedding)]
        mock_chunking = mock.MagicMock()
        mock_chunking.all_values.return_value = [
            {"method": "recursive", "chunk_size": 512, "chunk_overlap": 64},
        ]
        mock_retrieval = mock.MagicMock()
        mock_retrieval.all_values.return_value = [
            {"method": "simple", "number_of_chunks": 5, "search_mode": "vector"},
        ]

        mock_search_space = mock.MagicMock()
        mock_search_space.__getitem__ = mock.MagicMock(
            side_effect=lambda key: {
                "foundation_model": mock_foundation,
                "embedding_model": mock_embedding,
            }[key]
        )
        mock_search_space._search_space = {
            "foundation_model": mock_foundation,
            "embedding_model": mock_embedding,
            "chunking": mock_chunking,
            "retrieval": mock_retrieval,
        }
        return mock_search_space

    def test_report_includes_chunking_and_retrieval_blocks(self, tmp_path):
        """Report YAML contains chunking and retrieval search-space parameters."""
        mock_search_space = self._make_search_space()
        captured, _ = self._run_with_search_space(tmp_path, mock_search_space)

        assert captured["chunking"] == [{"method": "recursive", "chunk_size": 512, "chunk_overlap": 64}]
        assert captured["retrieval"] == [{"method": "simple", "number_of_chunks": 5, "search_mode": "vector"}]
        assert captured["foundation_model"] == ["gen-0"]
        assert captured["embedding_model"] == ["emb-0"]

    def test_models_preselector_limits_long_model_lists(self, tmp_path):
        """When model lists exceed TOP_N/TOP_K, ModelsPreSelector reduces selections."""
        mock_search_space = self._make_search_space(n_generation=5, n_embedding=4)

        import pandas as pd

        mocks = _make_all_mocks()
        benchmark_df = pd.DataFrame(
            [
                {
                    "question": "What is X?",
                    "correct_answers": [["Answer"]],
                    "correct_answer_document_ids": [["doc1"]],
                }
            ]
        )
        mock_pd = mock.MagicMock()
        mock_pd.read_json.return_value = benchmark_df
        mock_pd.DataFrame = pd.DataFrame
        mocks["pandas"] = mock_pd
        mocks["ogx_client"] = self._make_ogx_with_chat()
        mocks[
            "ai4rag.search_space.prepare.prepare_search_space"
        ].prepare_search_space_with_ogx.return_value = mock_search_space

        selected = {
            "foundation_models": ["gen-0", "gen-1", "gen-2"],
            "embedding_models": ["emb-0", "emb-1"],
        }
        mps_instance = mocks["ai4rag.core.experiment.mps"].ModelsPreSelector.return_value
        mps_instance.select_models.return_value = selected

        captured = {}

        def fake_safe_dump(data, stream):
            captured.update(data)

        mock_yaml = mock.MagicMock()
        mock_yaml.safe_dump = fake_safe_dump
        mocks["yaml"] = mock_yaml

        extracted_dir = tmp_path / "extracted"
        extracted_dir.mkdir()
        (extracted_dir / "doc0.md").write_text("sample text", encoding="utf-8")

        with (
            mock.patch.dict(sys.modules, mocks),
            mock.patch.dict(
                "os.environ",
                {"OGX_CLIENT_BASE_URL": "https://ogx.example.com", "OGX_CLIENT_API_KEY": "test-api-key"},
            ),
        ):
            search_space_preparation.python_func(
                test_data=mock.MagicMock(path=str(tmp_path / "test_data.json")),
                extracted_text=mock.MagicMock(path=str(extracted_dir)),
                search_space_prep_report=mock.MagicMock(path=str(tmp_path / "report.yml")),
                generation_models=[f"gen-{i}" for i in range(5)],
                embedding_models=[f"emb-{i}" for i in range(4)],
            )

        mps_cls = mocks["ai4rag.core.experiment.mps"].ModelsPreSelector
        mps_cls.assert_called_once()
        mps_instance.evaluate_patterns.assert_called_once()
        mps_instance.select_models.assert_called_once_with(n_embedding_models=2, n_foundation_models=3)

        assert captured["foundation_model"] == selected["foundation_models"]
        assert captured["embedding_model"] == selected["embedding_models"]
        assert len(captured["foundation_model"]) == 3
        assert len(captured["embedding_model"]) == 2

    def test_notebook_style_call_without_component_status(self):
        """Direct python_func calls without component_status work (notebook usage)."""
        import inspect

        # Verify component signature accepts component_status=None
        sig = inspect.signature(search_space_preparation.python_func)
        param = sig.parameters["component_status"]

        # Verify it has a default value of None
        assert param.default is None, "component_status should default to None for notebook usage"

        # This proves the component can be called without component_status
        # Full integration test would require real OGX/ai4rag dependencies
