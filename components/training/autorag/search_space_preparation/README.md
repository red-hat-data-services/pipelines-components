# Search Space Preparation 🔍

> ⚠️ **Stability: alpha** — This asset is not yet stable and may change.

## Overview 🧾

Builds and validates the search space of RAG configurations, including model preselection and
validation using in-memory vector databases.

The Search Space Preparation component (also known as Model Preselector) is a critical stage in the
AutoRAG pipeline that builds and validates the search space of RAG configurations. It validates
available models and their performance using an in-memory vector database, adjusting the search
space as needed. The component outputs a series of valid configurations and data for optimization.

This component uses the `ai4rag` library to systematically explore and validate RAG configuration
combinations based on provided constraints (chunking, embeddings, generation, retrieval). It
ensures that only valid and performant configurations are passed to the optimization stage,
reducing computational cost and improving optimization efficiency.

## Inputs 📥

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `constraints` | `dict` | `{}` | Dictionary containing search space constraints. See [Constraints Structure](#constraints-structure) below. |
| `models_config` | `dict` | `None` | Optional dictionary with models configuration. |
| `metric` | `str` | `faithfulness` | A RAG metric to optimise the experiment for. |

### Constraints Structure

The `constraints` dictionary should contain:

```python
{
    "chunking": [
        {
            "method": "recursive",
            "chunk_overlap": 256,
            "chunk_size": 2048
        }
    ],
    "embeddings": [
        {"model": "ibm/slate-125m-english-rtrvr-v2"},
        {"model": "intfloat/multilingual-e5-large"}
    ],
    "generation": [
        {"model": "mistralai/mixtral-8x7b-instruct-v01"},
        {"model": "ibm/granite-13b-instruct-v2"}
    ],
    "retrieval": [
        {
            "method": "simple",
            "number_of_chunks": 2,
            "hybrid_ranker": {
                "strategy": "weighted",
                "alpha": 0.6
            }
        }
    ]
}
```

## Outputs 📤

| Output | Type | Description |
|--------|------|-------------|
| `phase_report` | `dsl.OutputPath(dsl.Artifact)` | Path to a .yml-formatted file containing short report on the current experiment's phase. |

## Usage Examples 💡

### Basic Usage

```python
from kfp import dsl
from kfp_components.components.training.autorag.search_space_preparation import (
    search_space_preparation,
)

@dsl.pipeline(name="search-space-preparation-pipeline")
def my_pipeline():
    """Example pipeline for search space preparation."""
    constraints = {
        "chunking": [
            {
                "method": "recursive",
                "chunk_overlap": 256,
                "chunk_size": 2048
            }
        ],
        "embeddings": [
            {"model": "ibm/slate-125m-english-rtrvr-v2"}
        ],
        "generation": [
            {"model": "mistralai/mixtral-8x7b-instruct-v01"}
        ],
        "retrieval": [
            {
                "method": "simple",
                "number_of_chunks": 2
            }
        ]
    }
    
    prep_task = search_space_preparation(
        constraints=constraints
    )
    return prep_task
```

### With Multiple Constraints

```python
@dsl.pipeline(name="search-space-preparation-advanced-pipeline")
def my_pipeline():
    """Example pipeline with multiple constraint options."""
    constraints = {
        "chunking": [
            {
                "method": "recursive",
                "chunk_overlap": 256,
                "chunk_size": 2048
            },
            {
                "method": "recursive",
                "chunk_overlap": 128,
                "chunk_size": 1024
            }
        ],
        "embeddings": [
            {"model": "ibm/slate-125m-english-rtrvr-v2"},
            {"model": "intfloat/multilingual-e5-large"}
        ],
        "generation": [
            {"model": "mistralai/mixtral-8x7b-instruct-v01"},
            {
                "model": "ibm/granite-3-8b-instruct",
                "context_template_text": "\n[Document]\n{document}",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant."
                    }
                ]
            }
        ],
        "retrieval": [
            {
                "method": "simple",
                "number_of_chunks": 2,
                "hybrid_ranker": {
                    "strategy": "weighted",
                    "alpha": 0.6
                }
            },
            {
                "method": "simple",
                "number_of_chunks": 4
            }
        ]
    }
    
    prep_task = search_space_preparation(
        constraints=constraints
    )
    return prep_task
```

## Validation Process 🔧

The component performs the following validation steps:

1. **Model Availability**: Validates that specified models are available and accessible
2. **Configuration Compatibility**: Ensures configuration combinations are compatible
3. **Performance Testing**: Tests configurations using in-memory vector database
4. **Search Space Adjustment**: Adjusts the search space based on validation results
5. **Output Generation**: Produces validated configurations ready for optimization

## In-Memory Vector Database 🗄️

The component uses an in-memory vector database for:

- Model performance validation
- Configuration compatibility testing
- Search space optimization
- Early filtering of invalid configurations

## Notes 📝

- **Model Preselection**: Validates and preselects models based on performance
- **Search Space Reduction**: Filters out invalid configurations before optimization
- **ai4rag Integration**: Uses ai4rag library for systematic validation
- **Optimization Efficiency**: Reduces computational cost by validating configurations early

## Metadata 🗂️

- **Name**: search_space_preparation
- **Stability**: alpha
- **Dependencies**:
  - Kubeflow:
    - Name: Pipelines, Version: >=2.15.2
  - External Services:
    - Name: ai4rag, Version: >=1.0.0
- **Tags**:
  - training
  - autorag
  - search-space
  - optimization
- **Last Verified**: 2026-01-23 00:00:00+00:00

## Additional Resources 📚

- **AutoRAG Documentation**: See AutoRAG pipeline documentation for comprehensive information
- **ai4rag Documentation**: [ai4rag GitHub](https://github.com/IBM/ai4rag)
- **Issue Tracker**: [GitHub Issues](https://github.com/kubeflow/pipelines-components/issues)
