from typing import Optional

from kfp import dsl


@dsl.component(
    base_image="registry.redhat.io/rhoai/odh-pipeline-runtime-datascience-cpu-py312-rhel9@sha256:f9844dc150592a9f196283b3645dda92bd80dfdb3d467fa8725b10267ea5bdbc",
    packages_to_install=[
        "ai4rag@git+https://github.com/IBM/ai4rag.git",
        "pyyaml",
        "pysqlite3-binary",  # ChromaDB requires sqlite3 >= 3.35; base image has older sqlite
        "llama-stack-client",
        "openai",
    ],
)
def rag_templates_optimization(
    extracted_text: dsl.InputPath(dsl.Artifact),
    test_data: dsl.InputPath(dsl.Artifact),
    search_space_prep_report: dsl.InputPath(dsl.Artifact),
    rag_patterns: dsl.Output[dsl.Artifact],
    autorag_run_artifact: dsl.Output[dsl.Artifact],
    chat_model_url: Optional[str] = None,
    chat_model_token: Optional[str] = None,
    embedding_model_url: Optional[str] = None,
    embedding_model_token: Optional[str] = None,
    llama_stack_vector_database_id: Optional[str] = None,
    optimization_settings: Optional[dict] = None,
):
    """RAG Templates Optimization component.

    Carries out the iterative RAG optimization process.

    Args:
        extracted_text: A path pointing to a folder containg extracted texts from input documents.

        test_data: A path pointing to test data used for evaluating RAG pattern quality.

        search_space_prep_report: A path pointing to a .yml file containig short
            report on the experiment's first phase (search space preparation).

        rag_patterns: kfp-enforced argument specifying an output artifact. Provided by kfp backend automatically.

        autorag_run_artifact: kfp-enforced argument specifying an output artifact. Provided by kfp backend atomatically.

        chat_model_url: Inference endpoint URL for the chat/generation model (OpenAI-compatible).
            Required for in-memory scenario.

        chat_model_token: Optional API token for the chat model endpoint. Omit if deployment has no auth.

        embedding_model_url: Inference endpoint URL for the embedding model. Required for in-memory scenario.

        embedding_model_token: Optional API token for the embedding model endpoint. Omit if no auth.

        vector_database: An identificator of the vector store used in the experiment.

        llama_stack_vector_database_id: Vector database identifier as registered in llama-stack.

        optimization_settings: Additional settings customising the experiment.

    Returns:
        rag_patterns: Folder containing all generated RAG patterns (each subdir: pattern.json,
            indexing_notebook.ipynb, inference_notebook.ipynb).
        autorag_run_artifact: Run log and experiment status (TODO).
    """
    # ChromaDB (via ai4rag) requires sqlite3 >= 3.35; RHEL9 base image has older sqlite.
    # Patch stdlib sqlite3 with pysqlite3-binary before any ai4rag import.
    import sys

    try:
        import pysqlite3

        sys.modules["sqlite3"] = pysqlite3
    except ImportError:
        pass

    import os
    from collections import namedtuple
    from json import dump as json_dump
    from pathlib import Path

    import pandas as pd
    import yaml as yml
    from ai4rag.core.experiment.experiment import AI4RAGExperiment
    from ai4rag.core.experiment.results import ExperimentResults
    from ai4rag.core.hpo.gam_opt import GAMOptSettings
    from ai4rag.rag.embedding.base_model import BaseEmbeddingModel
    from ai4rag.rag.embedding.llama_stack import LSEmbeddingModel
    from ai4rag.rag.embedding.openai_model import OpenAIEmbeddingModel
    from ai4rag.rag.foundation_models.base_model import BaseFoundationModel
    from ai4rag.rag.foundation_models.llama_stack import LSFoundationModel
    from ai4rag.rag.foundation_models.openai_model import OpenAIFoundationModel
    from ai4rag.search_space.src.parameter import Parameter
    from ai4rag.search_space.src.search_space import AI4RAGSearchSpace
    from ai4rag.utils.event_handler.event_handler import BaseEventHandler, LogLevel
    from langchain_core.documents import Document
    from llama_stack_client import LlamaStackClient
    from openai import OpenAI

    MAX_NUMBER_OF_RAG_PATTERNS = 8
    METRIC = "faithfulness"
    SUPPORTED_OPTIMIZATION_METRICS = frozenset({"faithfulness", "answer_correctness", "context_correctness"})

    if embedding_model_url and chat_model_url:
        # Specification of OpenAI API compatibility
        embedding_model_url += "/v1"
        chat_model_url += "/v1"

    class TmpEventHandler(BaseEventHandler):
        """Exists temporarily only for the purpose of satisying type hinting checks"""

        def on_status_change(self, level: LogLevel, message: str, step: str | None = None) -> None:
            pass

        def on_pattern_creation(self, payload: dict, evaluation_results: list, **kwargs) -> None:
            pass

    def load_as_langchain_doc(path: str | Path) -> list[Document]:
        """Load a text file or folder into a list of langchain Document objects.

        Args:
            path: A local path to either a text file or a folder of text files.

        Returns:
            A list of langchain `Document` objects.
        """
        if isinstance(path, str):
            path = Path(path)

        documents = []
        if path.is_dir():
            for doc_path in path.iterdir():
                with doc_path.open("r", encoding="utf-8") as doc:
                    documents.append(Document(page_content=doc.read(), metadata={"document_id": doc_path.stem}))

        elif path.is_file():
            with path.open("r", encoding="utf-8") as doc:
                documents.append(Document(page_content=doc.read(), metadata={"document_id": path.stem}))

        return documents

    llama_stack_client_base_url = os.environ.get("LLAMA_STACK_CLIENT_BASE_URL", None)
    llama_stack_client_api_key = os.environ.get("LLAMA_STACK_CLIENT_API_KEY", None)

    in_memory_vector_store_scenario = False
    Client = namedtuple("Client", ["llama_stack", "generation_model", "embedding_model"], defaults=[None, None, None])

    if llama_stack_client_base_url and llama_stack_client_api_key:
        client = Client(llama_stack=LlamaStackClient())
    else:
        if not all((chat_model_url, chat_model_token, embedding_model_url, embedding_model_token)):
            raise ValueError(
                "All of (`chat_model_url`, `chat_model_token`, `embedding_model_url`, `embedding_model_token`) "
                "have to be defined when running AutoRAG experiment using an in-memory vector store."
            )
        client = Client(
            generation_model=OpenAI(api_key=chat_model_token, base_url=chat_model_url),
            embedding_model=OpenAI(api_key=embedding_model_token, base_url=embedding_model_url),
        )
        in_memory_vector_store_scenario = True

    def construct_model_instance(loader, node: yml.MappingNode) -> BaseEmbeddingModel | BaseFoundationModel:
        """Instructs yml.Loader on how to construct "!Model" tag."""
        mapping = loader.construct_mapping(node, deep=True)

        match mapping:
            case {"type_": "embedding", **id_to_params}:
                model_id, params = id_to_params.popitem()
                if in_memory_vector_store_scenario:
                    return OpenAIEmbeddingModel(client=client.embedding_model, model_id=model_id, params=params)
                else:
                    return LSEmbeddingModel(client=client.llama_stack, model_id=model_id, params=params)

            case {"type_": "generation", **id_to_params}:
                model_id, params = id_to_params.popitem()
                if in_memory_vector_store_scenario:
                    return OpenAIFoundationModel(client=client.generation_model, model_id=model_id, params=params)
                else:
                    return LSFoundationModel(client=client.llama_stack, model_id=model_id, params=params)
            case _:
                raise ValueError(f"Cannot load the yml-serialized !Model tag: {mapping}")

    yml.add_constructor("!Model", construct_model_instance, Loader=yml.SafeLoader)

    optimization_settings = optimization_settings if optimization_settings else {}
    if not (optimization_metric := optimization_settings.get("metric", None)):
        optimization_metric = METRIC
    if optimization_metric not in SUPPORTED_OPTIMIZATION_METRICS:
        raise ValueError(
            "optimization_metric must be one of %s; got %r"
            % (sorted(SUPPORTED_OPTIMIZATION_METRICS), optimization_metric)
        )

    documents = load_as_langchain_doc(extracted_text)

    # reload the search space
    with open(search_space_prep_report, "r") as f:
        search_space = yml.safe_load(f)

    search_space = AI4RAGSearchSpace(
        params=[Parameter(param, "C", values=values) for param, values in search_space.items()]
    )

    event_handler = TmpEventHandler()
    max_rag_patterns = optimization_settings.get("max_number_of_rag_patterns", MAX_NUMBER_OF_RAG_PATTERNS)
    optimizer_settings = GAMOptSettings(max_evals=int(max_rag_patterns))

    benchmark_data = pd.read_json(Path(test_data))

    if not llama_stack_vector_database_id and in_memory_vector_store_scenario:
        llama_stack_vector_database_id = "chroma"
    elif not llama_stack_vector_database_id:
        llama_stack_vector_database_id = "ls_milvus"

    rag_exp = AI4RAGExperiment(
        client=None if in_memory_vector_store_scenario else client.llama_stack,
        event_handler=event_handler,
        optimizer_settings=optimizer_settings,
        search_space=search_space,
        benchmark_data=benchmark_data,
        vector_store_type=llama_stack_vector_database_id,
        documents=documents,
        optimization_metric=optimization_metric,
        # TODO some necessary kwargs (if any at all)
    )

    # retrieve documents && run optimisation loop
    rag_exp.search()

    def _evaluation_result_fallback(eval_data_list, evaluation_result):
        """Build evaluation_results.json-style list when question_scores missing or incomplete."""
        out = []
        for ev in eval_data_list:
            answer_contexts = []
            if getattr(ev, "contexts", None) and getattr(ev, "context_ids", None):
                answer_contexts = [{"text": t, "document_id": doc_id} for t, doc_id in zip(ev.contexts, ev.context_ids)]
            scores = {}
            q_scores = (evaluation_result.scores or {}).get("question_scores") or {}
            for key in q_scores:
                if isinstance(q_scores[key], dict) and getattr(ev, "question_id", None) in q_scores[key]:
                    scores[key] = q_scores[key][ev.question_id]
            out.append(
                {
                    "question": getattr(ev, "question", ""),
                    "correct_answers": getattr(ev, "ground_truths", None),
                    "answer": getattr(ev, "answer", ""),
                    "answer_contexts": answer_contexts,
                    "scores": scores,
                }
            )
        return out

    rag_patterns_dir = Path(rag_patterns.path)
    evaluation_data_list = getattr(rag_exp.results, "evaluation_data", [])

    def _build_pattern_json(evaluation_result, iteration: int, max_combinations: int) -> dict:
        """Build pattern.json with flat schema (name, iteration, settings, scores, final_score)."""
        idx = evaluation_result.indexing_params or {}
        rp = evaluation_result.rag_params or {}
        chunking = idx.get("chunking") or {}
        # ai4rag puts embedding in indexing_params.embedding, not rag_params
        embedding_from_idx = idx.get("embedding") or idx.get("embeddings") or {}
        embeddings = rp.get("embeddings") or rp.get("embedding") or embedding_from_idx
        retrieval = rp.get("retrieval") or {}
        generation = rp.get("generation") or {}
        # embedding model_id: from indexing_params.embedding (ai4rag), or rag_params, or flat embedding_model
        embedding_model_id = None
        if isinstance(embedding_from_idx, dict) and embedding_from_idx.get("model_id"):
            embedding_model_id = embedding_from_idx.get("model_id")
        if not embedding_model_id and isinstance(embeddings, dict):
            embedding_model_id = embeddings.get("model_id")
        if not embedding_model_id and isinstance(rp.get("embedding_model"), str):
            embedding_model_id = rp.get("embedding_model")
        if not embedding_model_id and hasattr(rp.get("embedding_model"), "model_id"):
            embedding_model_id = getattr(rp.get("embedding_model"), "model_id", None)
        # generation model_id: from rag_params.generation (ai4rag) or flat foundation_model
        generation_model_id = generation.get("model_id") if isinstance(generation, dict) else None
        if not generation_model_id and isinstance(rp.get("foundation_model"), str):
            generation_model_id = rp.get("foundation_model")
        if not generation_model_id and hasattr(rp.get("foundation_model"), "model_id"):
            generation_model_id = getattr(rp.get("foundation_model"), "model_id", None)
        return {
            "name": getattr(evaluation_result, "pattern_name", ""),
            "iteration": iteration,
            "max_combinations": max_combinations,
            "duration_seconds": getattr(evaluation_result, "execution_time", 0) or 0,
            "settings": {
                "vector_store": {
                    "datasource_type": idx.get("vector_store", {}).get("datasource_type")
                    or rp.get("vector_store", {}).get("datasource_type")
                    or "ls_milvus",
                    "collection_name": getattr(evaluation_result, "collection", "") or "",
                },
                "chunking": {
                    "method": chunking.get("method", "recursive"),
                    "chunk_size": chunking.get("chunk_size", 2048),
                    "chunk_overlap": chunking.get("chunk_overlap", 256),
                },
                "embedding": {
                    "model_id": embedding_model_id or "",
                    "distance_metric": embeddings.get("distance_metric", "cosine")
                    if isinstance(embeddings, dict)
                    else "cosine",
                },
                "retrieval": {
                    "method": retrieval.get("method", "simple"),
                    "number_of_chunks": retrieval.get("number_of_chunks", 5),
                },
                "generation": {
                    "model_id": generation_model_id or "",
                    "context_template_text": generation.get("context_template_text", "{document}"),
                    "user_message_text": generation.get(
                        "user_message_text",
                        (
                            "\n\nContext:\n{reference_documents}:\n\nQuestion: {question}. \nAgain, please answer "
                            "the question based on the context provided only. If the context is not related to "
                            "the question, just say you cannot answer. Respond exclusively in the language of "
                            "the question."
                        ),
                    ),
                    "system_message_text": generation.get(
                        "system_message_text",
                        (
                            "Please answer the question I provide in the Question section below, based solely "
                            "on the information I provide in the Context section. If unanswerable, say so."
                        ),
                    ),
                },
            },
        }

    evaluations_list = list(rag_exp.results.evaluations)
    max_combinations = getattr(rag_exp.results, "max_combinations", len(evaluations_list)) or 24

    rag_patterns.metadata["name"] = "rag_patterns_artifact"
    rag_patterns.metadata["uri"] = rag_patterns.uri
    rag_patterns.metadata["metadata"] = {"patterns": []}
    for i, eval in enumerate(evaluations_list):
        patt_dir = rag_patterns_dir / eval.pattern_name
        patt_dir.mkdir(parents=True, exist_ok=True)

        pattern_data = _build_pattern_json(eval, iteration=i, max_combinations=max_combinations)
        # Flat schema: scores = per-metric aggregates (mean, ci_low, ci_high); final_score
        pattern_data["scores"] = (getattr(eval, "scores", None) or {}).get("scores") or {}
        pattern_data["final_score"] = getattr(eval, "final_score", None)
        rag_patterns.metadata["metadata"]["patterns"].append(pattern_data)
        with (patt_dir / "pattern.json").open("w+", encoding="utf-8") as pattern_details:
            json_dump(pattern_data, pattern_details, indent=2)

        with (patt_dir / "inference_notebook.ipynb").open("w+") as inf_notebook:
            json_dump({"inference_notebook_cell": "cell_value"}, inf_notebook)

        with (patt_dir / "indexing_notebook.ipynb").open("w+") as ind_notebook:
            json_dump({"ind_notebook_cell": "cell_Value"}, ind_notebook)

        eval_data = evaluation_data_list[i] if i < len(evaluation_data_list) else []
        try:
            q_scores = (eval.scores or {}).get("question_scores") or {}
            if q_scores and all(isinstance(q_scores.get(k), dict) for k in q_scores):
                evaluation_result_list = ExperimentResults.create_evaluation_results_json(eval_data, eval)
            else:
                evaluation_result_list = _evaluation_result_fallback(eval_data, eval)
        except (KeyError, TypeError):
            evaluation_result_list = _evaluation_result_fallback(eval_data, eval)
        with (patt_dir / "evaluation_results.json").open("w+", encoding="utf-8") as f:
            json_dump(evaluation_result_list, f, indent=2)

    # TODO autorag_run_artifact


if __name__ == "__main__":
    from kfp.compiler import Compiler

    Compiler().compile(
        rag_templates_optimization,
        package_path=__file__.replace(".py", "_component.yaml"),
    )
