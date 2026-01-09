# OSFT Training Pipeline

## Overview

A 4-stage pipeline for **Orthogonal Subspace Fine-Tuning (OSFT)** using the `mini-trainer` backend.

OSFT implements Orthogonal Subspace Fine-Tuning based on [Nayak et al. (2025), arXiv:2504.07097](https://arxiv.org/abs/2504.07097). This algorithm enables continual learning without catastrophic forgetting - no need for supplementary datasets to maintain original model distribution.

**Pipeline stages:**
1. **Dataset Download** - Fetch and validate training data from HuggingFace, S3, HTTP, or PVC
2. **OSFT Training** - Fine-tune using mini-trainer backend
3. **Evaluation** - Evaluate with lm-eval harness (arc_easy, mmlu, gsm8k, etc.)
4. **Model Registry** - Register trained model with full provenance tracking

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `phase_01_dataset_man_data_uri` | - | **[REQUIRED]** Dataset URI (e.g., `hf://dataset`, `s3://bucket/path`) |
| `phase_02_train_man_train_model` | `Qwen/Qwen2.5-1.5B-Instruct` | Base model to fine-tune |
| `phase_02_train_man_train_batch` | `128` | Effective batch size |
| `phase_02_train_man_train_epochs` | `1` | Training epochs |
| `phase_02_train_man_train_unfreeze` | `0.25` | Fraction to unfreeze (0.1=minimal, 0.5=strong) |
| `phase_02_train_man_train_tokens` | `64000` | Max tokens per GPU |
| `phase_02_train_man_train_workers` | `1` | Number of training pods |
| `phase_02_train_man_train_gpu` | `1` | GPUs per worker |
| `phase_02_train_opt_use_liger` | `True` | Enable Liger kernel optimizations |
| `phase_03_eval_man_eval_tasks` | `["arc_easy"]` | Evaluation benchmarks |
| `phase_04_registry_man_address` | `""` | Model Registry address (empty = skip) |

## Usage

```python
from kfp_components.pipelines.training.osft import osft_pipeline

# Compile the pipeline
from kfp import compiler
compiler.Compiler().compile(osft_pipeline, "osft_pipeline.yaml")
```

**Run with KFP:**
```bash
python -m pipelines.training.osft.pipeline  # generates pipeline.yaml
kfp pipeline upload -p "OSFT Training" pipelines/training/osft/pipeline.yaml
```

## Dataset Format

Training data should be a JSON Lines (.jsonl) file with messages format:

```json
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

## Required Kubernetes Secrets

| Secret | Keys | Purpose |
|--------|------|---------|
| `kubernetes-credentials` | `server_url`, `auth_token` | K8s API access for TrainJob |
| `minio-secret` (optional) | `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` | S3 dataset access |
| `hf-token` (optional) | `HF_TOKEN` | Gated model/dataset access |
