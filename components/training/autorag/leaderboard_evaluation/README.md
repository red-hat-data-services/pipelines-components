# Leaderboard Evaluation

> ⚠️ **Stability: alpha** — This asset is not yet stable and may change.

## Overview

Produces an HTML leaderboard artifact from RAG pattern evaluation results. It reads `pattern.json` from each pattern subdirectory (output of the RAG Templates Optimization component) and generates a single HTML table with RAG pattern names, settings, and metric values, ordered by final score (best first).

## Inputs

| Parameter     | Type                         | Description |
|--------------|------------------------------|-------------|
| `rag_patterns` | `dsl.InputPath[dsl.Artifact]` | Path to the directory of RAG patterns; each subdir must contain `pattern.json` (e.g. output of `rag_templates_optimization`). |

## Outputs

| Output        | Type                    | Description |
|---------------|-------------------------|-------------|
| `html_artifact` | `dsl.Output[dsl.HTML]` | HTML artifact (single file at artifact path) containing the leaderboard table (pattern name, settings, metrics, execution time, final score). Same output type and path semantics as the autogluon leaderboard_evaluation component. |

## Usage

Typically used downstream of `rag_templates_optimization` in the documents RAG optimization pipeline:

```python
from kfp import dsl
from kfp_components.components.training.autorag.leaderboard_evaluation import leaderboard_evaluation
from kfp_components.components.training.autorag.rag_templates_optimization import rag_templates_optimization

@dsl.pipeline(name="example")
def my_pipeline(...):
    opt_task = rag_templates_optimization(...)
    leaderboard_task = leaderboard_evaluation(
        rag_patterns=opt_task.outputs["rag_patterns"],
    )
```
