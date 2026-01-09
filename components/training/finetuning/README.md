# Finetuning Component

## Overview

Reusable training component for fine-tuning language models using Kubeflow Trainer and Training Hub. Supports both **SFT** (Supervised Fine-Tuning) and **OSFT** (Orthogonal Subspace Fine-Tuning) algorithms.

**Key capabilities:**
- Dual algorithm support (SFT/OSFT) via `training_algorithm` parameter
- Distributed training with configurable workers and GPUs
- OCI model registry support for base models
- Automatic checkpoint management and model artifact export

## Inputs

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pvc_path` | `str` | - | Workspace PVC root path (use `dsl.WORKSPACE_PATH_PLACEHOLDER`) |
| `dataset` | `dsl.Input[dsl.Dataset]` | `None` | Training dataset artifact |
| `training_base_model` | `str` | `Qwen/Qwen2.5-1.5B-Instruct` | Base model (HuggingFace ID, local path, or `oci://` ref) |
| `training_algorithm` | `str` | `OSFT` | Training algorithm: `OSFT` or `SFT` |
| `training_backend` | `str` | `mini-trainer` | Backend: `mini-trainer` (OSFT) or `instructlab-training` (SFT) |
| `training_effective_batch_size` | `int` | `128` | Effective batch size per optimizer step |
| `training_max_tokens_per_gpu` | `int` | `64000` | Max tokens per GPU (memory cap) |
| `training_num_epochs` | `int` | `1` | Number of training epochs |
| `training_learning_rate` | `float` | `5e-6` | Learning rate |
| `training_unfreeze_rank_ratio` | `float` | `0.25` | [OSFT] Fraction of parameters to unfreeze (0.1-0.5) |
| `training_resource_num_workers` | `int` | `1` | Number of training pods |
| `training_resource_gpu_per_worker` | `int` | `1` | GPUs per worker |

## Outputs

| Parameter | Type | Description |
|-----------|------|-------------|
| `output_model` | `dsl.Output[dsl.Model]` | Trained model checkpoint artifact |
| `output_metrics` | `dsl.Output[dsl.Metrics]` | Training metrics (loss, hyperparameters) |

## Usage

```python
from kfp_components.components.training.finetuning import train_model

# In your pipeline
training_task = train_model(
    pvc_path=dsl.WORKSPACE_PATH_PLACEHOLDER,
    dataset=dataset_task.outputs["train_dataset"],
    training_base_model="Qwen/Qwen2.5-1.5B-Instruct",
    training_algorithm="OSFT",  # or "SFT"
    training_effective_batch_size=128,
    training_num_epochs=1,
)
```
