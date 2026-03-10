# RAG Templates Optimization 🎯

> ⚠️ **Stability: alpha** — This asset is not yet stable and may change.

## Overview 🧾

Core optimization component that explores configurations to build optimized RAG Pattern(s) using
Generalized Additive Models (GAM).

The RAG Templates Optimization component is based on the `ai4rag` core optimization component that
explores configurations to build optimized RAG Pattern(s). It uses Generalized Additive Models
(GAM) to select the next configuration by predicting the evaluation score before execution. The
component uses Vector Database to create collection operations and retrieval requests during the
optimization process (supports both Milvus and Milvus Lite).

The component produces a leaderboard with RAG Patterns ranked by performance. Each RAG Pattern
includes validated parameter settings, performance metrics, and executable notebooks for
deployment.

## Inputs 📥

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `extracted_text` | `dsl.InputPath[dsl.Artifact]` | --- | Path to a file (or a folder of files) containing extracted texts from input documents. |
| `test_data` | `dsl.InputPath[dsl.Artifact]` | --- | Path to a .json file containing test data for evaluation. |
| `search_space_prep_report` | `dsl.InputPath[dsl.Artifact]` | --- | Path to a .yml file containing short report on the first experiment's phase (search space preparation). |
| `vector_database_id` | `str` | `None` | Optional vector database ID (e.g., registered in llama-stack Milvus database). If not provided, an in-memory database will be used. |
| `optimization_settings` | `dict` | `None` | Optional dictionary with optimization settings. See [Optimization Settings](#optimization-settings) below. |

### Optimization Settings

The `optimization_settings` dictionary supports:

```python
{
    "max_number_of_rag_patterns": 4,      # Maximum number of RAG patterns to generate
    "metric": "answer_correctness"         # Metric to optimize: Literal["answer_correctness", "faithfulness", "context_correctness"] 
}
```

**Supported Metrics:**

- `"answer_correctness"` - Measures the correctness of generated answers
- `"faithfulness"` - Measures how faithful the answer is to the retrieved context
- `"context_correctness"` - Automatically calculated, measures the quality of retrieved chunks

## Outputs 📤

| Output | Type | Description |
|--------|------|-------------|
| `rag_patterns` | `dsl.Output[dsl.Artifact]` | Directory of RAG Pattern artifacts; each subdir contains `pattern.json`, `evaluation_results.json`, `indexing_notebook.ipynb`, and `inference_notebook.ipynb`. Consumed by the **leaderboard_evaluation** component to produce the HTML leaderboard. |
| `autorag_run_artifact` | `dsl.Output[dsl.Artifact]` | General type artifact pointing to the log file and an experiment status object. |

## Usage Examples 💡

### Basic Usage

```python
from kfp import dsl
from kfp_components.components.training.autorag.rag_templates_optimization import (
    rag_templates_optimization,
)

@dsl.pipeline(name="rag-optimization-pipeline")
def my_pipeline(extracted_text, test_data, search_space_prep_report):
    """Example pipeline for RAG optimization."""
    opt_task = rag_templates_optimization(
        extracted_text=extracted_text,
        test_data=test_data,
        search_space_prep_report=search_space_prep_report,
        optimization_settings={
            "max_number_of_rag_patterns": 4,
            "metric": "answer_correctness"
        }
    )
    return opt_task
```

### With Vector Database

```python
@dsl.pipeline(name="rag-optimization-with-vector-db-pipeline")
def my_pipeline(extracted_text, test_data, search_space_prep_report):
    """Example pipeline with persistent vector database."""
    opt_task = rag_templates_optimization(
        extracted_text=extracted_text,
        test_data=test_data,
        search_space_prep_report=search_space_prep_report,
        vector_database_id="milvus-database",
        optimization_settings={
            "max_number_of_rag_patterns": 4,
            "metric": "faithfulness"
        }
    )
    return opt_task
```

## GAM-Based Optimization 🎯

The component uses Generalized Additive Models (GAM) for intelligent configuration selection:

1. **Score Prediction**: Predicts evaluation scores for configurations before execution
2. **Configuration Selection**: Selects the most promising configurations based on predictions
3. **Iterative Exploration**: Systematically explores the search space
4. **Performance Ranking**: Ranks configurations by actual performance metrics

This approach significantly reduces the number of configurations that need to be evaluated,
improving optimization efficiency.

## Vector Database Integration 🗄️

The component integrates with vector databases for:

- **Collection Operations**: Creates and manages vector collections
- **Retrieval Requests**: Performs retrieval operations during optimization
- **Index Management**: Handles vector index creation and updates

**Supported Databases:**

- **Milvus**: Persistent vector database (requires `vector_database_id`)
- **Milvus Lite**: In-memory vector database (used when `vector_database_id` is not provided)

## RAG Patterns Output 📦

Each RAG Pattern artifact includes:

- **Optimized Configuration**: Validated parameter settings (chunking, embeddings, generation,
  retrieval)
- **Performance Metrics**: answer_correctness, faithfulness, context_correctness
- **Index Building Notebook**: For building vector index/collection
- **Retrieval/Generation Notebook**: For performing retrieval and generation operations
- **Leaderboard Position**: Ranking based on performance

## Notes 📝

- **ai4rag Core**: Based on ai4rag core optimization component
- **GAM Prediction**: Uses GAM to predict scores before execution, reducing evaluation overhead
- **Vector Database**: Supports both persistent (Milvus) and in-memory (Milvus Lite) databases
- **Multiple Patterns**: Can generate multiple RAG Patterns ranked by performance
- **Production Ready**: RAG Patterns are ready for deployment with executable notebooks

## Metadata 🗂️

- **Name**: rag_templates_optimization
- **Stability**: alpha
- **Dependencies**:
  - Kubeflow:
    - Name: Pipelines, Version: >=2.15.2
  - External Services:
    - Name: ai4rag, Version: >=1.0.0
    - Name: llama-stack API, Version: >=1.0.0
    - Name: Milvus, Version: >=2.0.0 (optional)
    - Name: Milvus Lite, Version: >=2.0.0
- **Tags**:
  - training
  - autorag
  - optimization
  - rag-patterns
- **Last Verified**: 2026-01-23 00:00:00+00:00

## Additional Resources 📚

- **AutoRAG Documentation**: See AutoRAG pipeline documentation for comprehensive information
- **ai4rag Documentation**: [ai4rag GitHub](https://github.com/IBM/ai4rag)
- **Issue Tracker**: [GitHub Issues](https://github.com/kubeflow/pipelines-components/issues)
