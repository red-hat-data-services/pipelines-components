# SFT Training Pipeline

## Overview

A 4-stage pipeline for **Supervised Fine-Tuning (SFT)** using the `instructlab-training` backend with PyTorch FSDP for distributed training.

**Pipeline stages:**
1. **Dataset Download** - Fetch and validate training data from HuggingFace, S3, HTTP, or PVC
2. **SFT Training** - Fine-tune using instructlab-training backend
3. **Evaluation** - Evaluate with lm-eval harness (arc_easy, mmlu, gsm8k, etc.)
4. **Model Registry** - Register trained model with full provenance tracking

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `phase_01_dataset_man_data_uri` | - | **[REQUIRED]** Dataset URI (e.g., `hf://dataset`, `s3://bucket/path`) |
| `phase_02_train_man_model` | `Qwen/Qwen2.5-1.5B-Instruct` | Base model to fine-tune |
| `phase_02_train_man_batch` | `128` | Effective batch size |
| `phase_02_train_man_epochs` | `1` | Training epochs |
| `phase_02_train_man_tokens` | `10000` | Max tokens per GPU |
| `phase_02_train_man_workers` | `4` | Number of training pods |
| `phase_02_train_man_gpu` | `1` | GPUs per worker |
| `phase_02_train_opt_fsdp_sharding` | `FULL_SHARD` | FSDP sharding strategy |
| `phase_03_eval_man_tasks` | `["arc_easy"]` | Evaluation benchmarks |
| `phase_04_registry_man_address` | `""` | Model Registry address (empty = skip) |

## Usage

```python
from kfp_components.pipelines.training.sft import sft_pipeline

# Compile the pipeline
from kfp import compiler
compiler.Compiler().compile(sft_pipeline, "sft_pipeline.yaml")
```

**Run with KFP:**
```bash
python -m pipelines.training.sft.pipeline  # generates pipeline.yaml
kfp pipeline upload -p "SFT Training" pipelines/training/sft/pipeline.yaml
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
