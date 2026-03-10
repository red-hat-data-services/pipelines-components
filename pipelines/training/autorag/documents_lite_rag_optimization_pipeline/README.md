# Documents Lite RAG Optimization Pipeline ✨

> ⚠️ **Stability: alpha** — This asset is not yet stable and may change.

## Overview 🧾

The Documents Lite RAG Optimization Pipeline is an automated system for building and optimizing
Retrieval-Augmented Generation (RAG) applications within Red Hat OpenShift AI. It leverages
Kubeflow Pipelines to orchestrate the optimization workflow, using the ai4rag optimization
engine to systematically explore RAG configurations and identify the best performing parameter
settings based on an upfront-specified quality metric.

This **lite** variant uses an in-memory ChromaDB vector store and OpenAI-compatible clients to
talk to chat and embedding models directly (via `chat_model_url` / `embedding_model_url` and
tokens). It does not use llama-stack API or external vector databases.

The optimization process involves the following stages:

1. **Test Data Loading**: Loads test data from JSON files for evaluation
2. **Document Loading & Sampling**: Loads documents from data sources and chooses a subset based on the test data
3. **Text Extraction**: Extracts text from sampled documents using the docling library
4. **Search Space Preparation**: Builds the search space defining RAG configurations to try out. Limits foundation and embedding models; uses in-memory ChromaDB for validation.
5. **RAG Templates Optimization**: Systematically tests different RAG configurations from the defined search space using GAM-based prediction. Produces artifacts including RAG Patterns, associated metrics, logs and notebooks.
6. **Evaluation**: Assesses each configuration's performance using test data
7. **Leaderboard**: Builds a leaderboard of RAG Patterns ranked by performance

## Inputs 📥

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `test_data_secret_name` | `str` | — | Kubernetes secret name for S3-compatible credentials (test data). Must provide: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_S3_ENDPOINT, AWS_DEFAULT_REGION. |
| `test_data_bucket_name` | `str` | — | S3 (or compatible) bucket name for the test data JSON file. |
| `test_data_key` | `str` | — | Object key (path) of the test data JSON file in the test data bucket. |
| `input_data_secret_name` | `str` | — | Kubernetes secret name for S3-compatible credentials (input documents). Same env vars as above. |
| `input_data_bucket_name` | `str` | — | S3 (or compatible) bucket name for the input documents. |
| `input_data_key` | `str` | — | Object key (path) of the input documents in the input data bucket. |
| `chat_model_url` | `str` | — | Inference endpoint URL for the chat/generation model (OpenAI-compatible). |
| `chat_model_token` | `str` | — | API token or key for authenticating with the chat model endpoint. |
| `embedding_model_url` | `str` | — | Inference endpoint URL for the embedding model. |
| `embedding_model_token` | `str` | — | API token or key for authenticating with the embedding model endpoint. |
| `optimization_metric` | `str` | `"faithfulness"` | Metric to optimize. Supported: `faithfulness`, `answer_correctness`, `context_correctness`. |

## Outputs 📤

| Output | Type | Description |
| ------ | ---- | ----------- |
| `documents_sampling_task.sampled_documents_atrifact` | `dsl.Output[dsl.Artifact]` | JSON artifact containing sampled documents metadata. |
| `text_extraction_task.extracted_text_artifact` | `dsl.Output[dsl.Artifact]` | Extracted text from documents (folder with markdown files). |
| `rag_templates_optimization.rag_patterns_artifact` | `dsl.Output[dsl.Artifact]` | A directory containing one subdirectory per top-N RAG pattern (named by pattern). In this lite pipeline the vector store used during optimization is ChromaDB in-memory. |
| `leaderboard_evaluation_task.html_artifact` | `dsl.Output[dsl.HTML]` | HTML leaderboard table of RAG patterns ranked by `final_score`, with pattern name, metrics (e.g. mean_answer_correctness, mean_faithfulness, mean_context_correctness), config columns (chunking, embedding, retrieval, generation), and links to indexing/inference notebooks. |

### `rag_patterns_artifact` details

Each RAG Pattern subdirectory includes:

- **pattern.json** — Flat schema with `name`, `iteration`, `max_combinations`, `duration_seconds`, `settings` (vector_store, chunking, embedding, retrieval, generation), `scores` (per-metric mean/ci_low/ci_high), and `final_score`. In this lite pipeline the vector store is ChromaDB in-memory.
- **evaluation_results.json** — Per-question evaluation (question, answer, correct_answers, answer_contexts, scores)
- **indexing_notebook.ipynb** — Notebook for building or populating the vector index/collection
- **inference_notebook.ipynb** — Notebook for retrieval and generation

#### Metadata of artifact

```json
{
   "name":"rag_patterns_artifact",
   "uri":"documents-lite-rag-optimization-pipeline/<run_id>/rag-templates-optimization/<task_id>/rag_patterns/",
   "metadata":{
      "patterns":[
         {
            "name":"pattern0",
            "iteration":0,
            "max_combinations":3,
            "duration_seconds":20,
            "location": {
               "evaluation_results": "pattern0/evaluation_results.json",
               "indexing_notebook": "pattern0/indexing_notebook.ipynb",
               "inference_notebook": "pattern0/inference_notebook.ipynb",
               "pattern_descriptor": "pattern0/pattern.json"
            },
            "settings":{
               "vector_store":{
                  "datasource_type":"chroma",
                  "collection_name":"collection0"
               },
               "chunking":{
                  "method":"recursive",
                  "chunk_size":256,
                  "chunk_overlap":128
               },
               "embedding":{
                  "model_id":"mock-embed-a",
                  "distance_metric":"cosine"
               },
               "retrieval":{
                  "method":"window",
                  "number_of_chunks":5
               },
               "generation":{
                  "model_id":"mock-llm-1",
                  "context_template_text":"{document}",
                  "user_message_text":"<prompt: context + question; answer in question language>",
                  "system_message_text":"<system: answer from context only; say if unanswerable>"
               }
            },
            "scores":{
               "answer_correctness":{"mean":0.5,"ci_low":0.4,"ci_high":0.7},
               "faithfulness":{"mean":0.2,"ci_low":0.1,"ci_high":0.5},
               "context_correctness":{"mean":1.0,"ci_low":0.9,"ci_high":1.0}
            },
            "final_score":0.5
         }
      ]
   }
}
```

#### Evaluation results

Structure matches ai4rag `ExperimentResults.create_evaluation_results_json()`. For details see the sample below.

```json
[
  {
    "question": "What foundation models are available in watsonx.ai?",
    "correct_answers": ["The following models are available in watsonx.ai: flan-t5-xl-3b, ..."],
    "answer": "Watsonx.ai provides foundation models such as flan-t5-xl-3b, granite-13b-instruct-v2, and others.",
    "answer_contexts": [
      { "text": "Model architecture influences how the model behaves.", "document_id": "120CAE8361AE4E0B6FE4D6F0D32EEE9517F11190_1.txt" },
      { "text": "Learn more about governing assets in AI use cases.", "document_id": "0ECEAC44DA213D067B5B5EA66694E6283457A441_9.txt" }
    ],
    "scores": {
      "answer_correctness": 0.72,
      "faithfulness": 0.85,
      "context_correctness": 1.0
    }
  },
  {
    "question": "How can I ensure generated answers are accurate and factual?",
    "correct_answers": ["Utilize RAG, prompt engineering, and validate output."],
    "answer": "Use retrieval-augmented generation and validate the model output against your data.",
    "answer_contexts": [
      { "text": "Retrieval-augmented generation in IBM watsonx.ai.", "document_id": "752D982C2F694FFEE2A312CEA6ADF22C2384D4B2_0.txt" }
    ],
    "scores": {
      "answer_correctness": 0.65,
      "faithfulness": 0.91,
      "context_correctness": 0.8
    }
  }
]
```

#### Pattern details

Collection of pattern details and related metadata. For details see the sample pattern.json below.

```json
{
  "name": "Pattern_0",
  "iteration": 0,
  "max_combinations": 390,
  "duration_seconds": 100,
  "settings": {
    "vector_store": {
      "datasource_type": "chroma",
      "collection_name": "collection0"
    },
    "chunking": {
      "method": "recursive",
      "chunk_size": 256,
      "chunk_overlap": 128
    },
    "embedding": {
      "model_id": "mock-embed-a",
      "distance_metric": "cosine"
    },
    "retrieval": {
      "method": "recursive",
      "number_of_chunks": 5
    },
    "generation": {
      "model_id": "mock-llm-1",
      "context_template_text": "{document}",
      "user_message_text": "<prompt template: context + question; answer in question language>",
      "system_message_text": "<system prompt: answer from context only; say if unanswerable>"
    }
  },
  "scores": {
    "answer_correctness": { "mean": 0.5, "ci_low": 0.4, "ci_high": 0.7 },
    "faithfulness": { "mean": 0.2, "ci_low": 0.1, "ci_high": 0.5 },
    "context_correctness": { "mean": 1.0, "ci_low": 0.9, "ci_high": 1.0 }
  },
  "final_score": 0.5
}
```

### Files stored in user storage

Pipeline outputs are written to the artifact store (S3-compatible storage configured for Kubeflow Pipelines). The layout below matches what components write and what downstream consumers expect when loading the leaderboard or a refitted model.

```text
<pipeline_name>/
└── <run_id>/
    ├── documents-sampling/
    │   └── <task_id>/
    │       └── sampled_documents_atrifact        # YAML artifact containing sampled documents metadata
    ├── text-extraction/
    │   └── <task_id>/
    │       └── extracted_text_artifact           # Folder containing markdown files with extracted text
    ├── leaderboard-evaluation/
    │   └── <task_id>/
    │       └── html_artifact                     # HTML leaderboard (RAG pattern names + metrics); single file at path
    └── rag-templates-optimization/
        └── <task_id>/
            └── autorag_run_artifact              # Log and experiment status
            └── rag_patterns_artifact/
                ├── <pattern_name_0>/             # One per top-N RAG pattern
                │   ├── pattern.json              # Flat schema: name, iteration, settings, scores, final_score
                │   ├── evaluation_results.json   # Per-question evaluation (question, answer, correct_answers, scores, etc.)
                │   ├── indexing_notebook.ipynb   # Notebook to build/populate the vector index
                │   └── inference_notebook.ipynb  # Notebook for retrieval and generation
                ├── <pattern_name_1>/
                │   ├── pattern.json
                │   ├── evaluation_results.json
                │   ├── indexing_notebook.ipynb
                │   └── inference_notebook.ipynb
                └── ...
```

- `pipeline_name`: pipeline identifier (e.g. `documents-lite-rag-optimization-pipeline`).
- `run_id`: Kubeflow Pipelines run ID.
- Component folders align with pipeline steps; `<task_id>` is the KFP task ID for that step.
- Pattern count and names depend on the run (e.g. `max_number_of_rag_patterns`).

## Example usage

```python
from kfp_components.pipelines.training.autorag.documents_lite_rag_optimization_pipeline import (
    documents_lite_rag_optimization_pipeline,
)


def my_pipeline():
    """Full example with optional parameters."""
    return documents_lite_rag_optimization_pipeline(
        test_data_secret_name="s3-test-data-secret",
        test_data_bucket_name="autorag-benchmarks",
        test_data_key="my-folder/test_data.json",
        input_data_secret_name="s3-input-secret",
        input_data_bucket_name="my-documents-bucket",
        input_data_key="rh_documents/",
        chat_model_url="https://my-chat-api.example.com/v1",
        chat_model_token="sk-...",
        embedding_model_url="https://my-embedding-api.example.com/v1",
        embedding_model_token="sk-...",
        optimization_metric="answer_correctness",
    )
```

<!-- custom-content -->

## Optimization Engine: ai4rag 🚀

The pipeline uses [ai4rag](https://github.com/IBM/ai4rag), a RAG Templates Optimization Engine that
provides an automated approach to optimizing Retrieval-Augmented Generation (RAG) systems. The
engine is designed to be LLM and Vector Database provider agnostic, making it flexible and
adaptable to various RAG implementations.

ai4rag accepts a variety of RAG templates and search space definitions, then systematically
explores different parameter configurations to find optimal settings. The engine returns initialized
RAG templates with optimal parameter values, which are referred to as RAG Patterns.

## Supported Features ✨

**Status**: Tech Preview - MVP (May 2026)

### RAG Configuration

- **RAG Type**: Documents (documents provided as input)
- **Supported Languages**: English
- **Supported Document Types**: PDF, DOCX, PPTX, Markdown, HTML, Plain text
- **Document Data Sources**: S3 (Amazon S3), Local filesystem (FS)

### Infrastructure Components

- **Vector store**: ChromaDB (in-memory), used during optimization
- **Chat and embedding**: OpenAI-compatible API; endpoints and tokens provided via pipeline
  parameters (`chat_model_url`, `chat_model_token`, `embedding_model_url`, `embedding_model_token`)

### Processing Methods

- **Chunking Method**: Recursive
- **Retrieval Methods**: Simple, Simple with hybrid ranker

### Interfaces

- **API**: Programmatic access to AutoRAG functionality
- **UI**: User interface for interacting with AutoRAG

## Glossary 📚

### RAG Configuration Definition

A **RAG Configuration** is a specific set of parameter values that define how a
Retrieval-Augmented Generation system operates. It includes settings for:

- **Chunking**: Method and parameters for splitting documents (e.g., recursive method with
  chunk_size=2048, chunk_overlap=256)
- **Embeddings**: The embedding model used (e.g., `intfloat/multilingual-e5-large`)
- **Generation**: The language model used (e.g., `ibm/granite-13b-instruct-v2`) along with its
  parameters
- **Retrieval**: The method for retrieving relevant document chunks (e.g., simple retrieval or
  hybrid ranker)

### RAG Pattern

A **RAG Pattern** is an optimized RAG configuration that has been evaluated and ranked by
AutoRAG. It represents a complete, deployable RAG system with:

- Validated parameter settings that have been tested and evaluated
- Performance metrics (e.g., answer_correctness, faithfulness, context_correctness)
- Executable notebooks for indexing and inference operations
- A position in the leaderboard based on performance

### RAG Template

A **RAG Template** is a reusable blueprint that defines the structure and workflow of a RAG
system. Templates are parameterized and AutoRAG uses templates as the foundation, optimizing the
parameter values to create RAG Patterns.

## Additional Resources 📚

- **ai4rag Documentation**: [ai4rag GitHub](https://github.com/IBM/ai4rag)
- **Issue Tracker**: [GitHub Issues](https://github.com/kubeflow/pipelines-components/issues)

## Metadata 🗂️

- **Name**: documents_lite_rag_optimization_pipeline
- **Stability**: alpha
- **Dependencies**:
  - Kubeflow:
    - Name: Pipelines, Version: >=2.15.2
  - External Services:
    - Name: ai4rag, Version: >=1.0.0
    - Name: docling, Version: >=1.0.0
  - Runtime:
    - In-memory vector store: ChromaDB (via ai4rag)
    - Chat and embedding: OpenAI-compatible API (URL + token)
- **Tags**:
  - training
  - pipeline
  - autorag
  - rag-optimization
- **Last Verified**: 2026-01-23 14:57:32+00:00
- **Owners**:
  - Approvers: None
  - Reviewers: None
