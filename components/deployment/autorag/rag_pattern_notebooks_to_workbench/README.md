# RAG Pattern Notebooks to Workbench 📓

> ⚠️ **Stability: alpha** — This asset is not yet stable and may change.

## Overview 🧾

Deploys RAG Pattern notebooks to Red Hat OpenShift AI Workbench for interactive use.

The RAG Pattern Notebooks to Workbench component takes optimized RAG Pattern artifacts from the
optimization stage and deploys them to Red Hat OpenShift AI Workbench. This enables users to
interactively work with the optimized RAG Patterns, including building vector indexes and
performing retrieval and generation operations.

Each RAG Pattern includes two notebooks:

1. **Index Building Notebook**: For building the vector index/collection or populating existing
   indexes with documents
2. **Retrieval/Generation Notebook**: For performing retrieval and generation operations

## Inputs 📥

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `workbench_notebooks` | `dsl.Output[dsl.Artifact]` | `None` | Output artifact containing information about deployed notebooks in Workbench. |
| `rag_patterns` | `dsl.Input[dsl.Artifact]` | `None` | Input artifact(s) containing RAG Pattern(s) with notebooks from optimization stage. |
| `workbench_config` | `dict` | `None` | Optional dictionary with Workbench configuration. See [Workbench Configuration](#workbench-configuration) below. |

### Workbench Configuration

The `workbench_config` dictionary supports:

```python
{
    "workbench_id": "my-workbench",        # Workbench ID or name
    "namespace": "default",                # Kubernetes namespace
    "notebook_path": "/notebooks/autorag"  # Path within workbench for notebooks
}
```

## Outputs 📤

| Output | Type | Description |
|--------|------|-------------|
| `workbench_notebooks` | `dsl.Artifact` | Artifact containing information about deployed notebooks, including paths and access information. |
| Return value | `str` | A message indicating the completion status of notebook deployment. |

## Usage Examples 💡

### Basic Usage

```python
from kfp import dsl
from kfp_components.components.deployment.autorag.rag_pattern_notebooks_to_workbench import (
    rag_pattern_notebooks_to_workbench,
)

@dsl.pipeline(name="rag-pattern-deployment-pipeline")
def my_pipeline(rag_patterns):
    """Example pipeline for deploying RAG Pattern notebooks."""
    deploy_task = rag_pattern_notebooks_to_workbench(
        rag_patterns=rag_patterns,
        workbench_config={
            "workbench_id": "my-workbench",
            "namespace": "default"
        }
    )
    return deploy_task
```

### With Custom Path

```python
@dsl.pipeline(name="rag-pattern-deployment-custom-path-pipeline")
def my_pipeline(rag_patterns):
    """Example pipeline with custom notebook path."""
    deploy_task = rag_pattern_notebooks_to_workbench(
        rag_patterns=rag_patterns,
        workbench_config={
            "workbench_id": "my-workbench",
            "namespace": "default",
            "notebook_path": "/notebooks/autorag/patterns"
        }
    )
    return deploy_task
```

## RAG Pattern Notebooks 📓

Each RAG Pattern includes two notebooks:

### Index Building Notebook

The index building notebook is used for:

- **In-Memory Database**: Building the vector index/collection from scratch
- **Persistent Database**: Populating already existing index/collection (created during experiment)
  with all user documents
- **Document Processing**: Handling document sampling and indexing operations

> 📘 **Note**: When dealing with persistent vector stores, remember that documents were sampled at
> the very start of the experiment. The notebook populates the existing index with all of the
> user's documents.

### Retrieval/Generation Notebook

The retrieval/generation notebook is used for:

- **Retrieval Operations**: Performing document retrieval using the optimized configuration
- **Generation Operations**: Generating answers using the optimized LLM configuration
- **Interactive Testing**: Testing the RAG Pattern with custom queries
- **Performance Evaluation**: Evaluating the RAG Pattern on custom test sets

## Workbench Integration 🔧

The component integrates with Red Hat OpenShift AI Workbench to:

1. **Notebook Deployment**: Deploys notebooks to the specified Workbench instance
2. **Path Management**: Organizes notebooks in the specified directory structure
3. **Access Configuration**: Configures notebook access and permissions
4. **Metadata Storage**: Stores deployment information for tracking

## Notes 📝

- **Interactive Access**: Enables interactive use of optimized RAG Patterns in Workbench
- **Notebook Execution**: Users can execute notebooks directly in Workbench environment
- **Pattern Deployment**: Deploys all notebooks from RAG Pattern artifacts
- **Workbench Integration**: Seamlessly integrates with Red Hat OpenShift AI Workbench

## Metadata 🗂️

- **Name**: rag_pattern_notebooks_to_workbench
- **Stability**: alpha
- **Dependencies**:
  - Kubeflow:
    - Name: Pipelines, Version: >=2.15.2
  - External Services:
    - Name: Red Hat OpenShift AI Workbench, Version: >=1.0.0
- **Tags**:
  - deployment
  - autorag
  - workbench
  - notebooks
- **Last Verified**: 2026-01-23 00:00:00+00:00

## Additional Resources 📚

- **AutoRAG Documentation**: See AutoRAG pipeline documentation for comprehensive information
- **Issue Tracker**: [GitHub Issues](https://github.com/kubeflow/pipelines-components/issues)
