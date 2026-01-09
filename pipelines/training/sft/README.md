# Sft Pipeline ✨

## Overview 🧾

SFT Training Pipeline - Standard supervised fine-tuning with instructlab-training.

A 4-stage ML pipeline for fine-tuning language models:
1) Dataset Download - Prepares training data from HuggingFace, S3, HTTP, or PVC
2) SFT Training - Fine-tunes using instructlab-training backend
3) Evaluation - Evaluates with lm-eval harness (MMLU, GSM8K, etc.)
4) Model Registry - Registers trained model to Kubeflow Model Registry

## Inputs 📥

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `phase_01_dataset_man_data_uri` | `str` | `None` | [REQUIRED] Dataset location (hf://dataset, s3://bucket/path, https://url, pvc://path) |
| `phase_01_dataset_man_data_split` | `float` | `0.9` | Train/eval split ratio (0.9 = 90% train, 10% eval) |
| `phase_02_train_man_batch` | `int` | `128` | Effective batch size (samples per optimizer step). Start with 128 |
| `phase_02_train_man_epochs` | `int` | `1` | Number of training epochs. 1 is often sufficient |
| `phase_02_train_man_gpu` | `int` | `1` | GPUs per worker. KEEP AT 1 to avoid /dev/shm issues |
| `phase_02_train_man_model` | `str` | `Qwen/Qwen2.5-1.5B-Instruct` | Base model (HuggingFace ID or path) |
| `phase_02_train_man_tokens` | `int` | `10000` | Max tokens per GPU (memory cap). 10000 for SFT |
| `phase_02_train_man_workers` | `int` | `4` | Number of training pods. 4 pods x 1 GPU = 4 total GPUs |
| `phase_03_eval_man_tasks` | `list` | `['arc_easy']` | lm-eval tasks (arc_easy, mmlu, gsm8k, hellaswag, etc.) |
| `phase_04_registry_man_address` | `str` | `` | Model Registry address (empty = skip registration) |
| `phase_04_registry_man_author` | `str` | `pipeline` | Author name for the registered model |
| `phase_04_registry_man_name` | `str` | `sft-model` | Model name in registry |
| `phase_04_registry_man_version` | `str` | `1.0.0` | Semantic version (major.minor.patch) |
| `phase_01_dataset_opt_hf_token` | `str` | `` |  |
| `phase_01_dataset_opt_subset` | `int` | `0` |  |
| `phase_02_train_opt_annotations` | `str` | `` |  |
| `phase_02_train_opt_cpu` | `str` | `4` |  |
| `phase_02_train_opt_env_vars` | `str` | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,NCCL_DEBUG=INFO,INSTRUCTLAB_NCCL_TIMEOUT_MS=600000` |  |
| `phase_02_train_opt_hf_token` | `str` | `` |  |
| `phase_02_train_opt_labels` | `str` | `` |  |
| `phase_02_train_opt_learning_rate` | `float` | `5e-06` |  |
| `phase_02_train_opt_lr_warmup` | `int` | `0` |  |
| `phase_02_train_opt_lr_scheduler` | `str` | `cosine` |  |
| `phase_02_train_opt_max_seq_len` | `int` | `8192` |  |
| `phase_02_train_opt_memory` | `str` | `64Gi` |  |
| `phase_02_train_opt_num_procs` | `str` | `auto` |  |
| `phase_02_train_opt_pull_secret` | `str` | `` |  |
| `phase_02_train_opt_save_epoch` | `bool` | `True` |  |
| `phase_02_train_opt_save_full_state` | `bool` | `False` |  |
| `phase_02_train_opt_fsdp_sharding` | `str` | `FULL_SHARD` |  |
| `phase_02_train_opt_save_samples` | `int` | `0` |  |
| `phase_02_train_opt_seed` | `int` | `42` |  |
| `phase_02_train_opt_use_liger` | `bool` | `False` |  |
| `phase_03_eval_opt_batch` | `str` | `auto` |  |
| `phase_03_eval_opt_gen_kwargs` | `dict` | `{}` |  |
| `phase_03_eval_opt_limit` | `int` | `-1` |  |
| `phase_03_eval_opt_log_samples` | `bool` | `True` |  |
| `phase_03_eval_opt_model_args` | `dict` | `{}` |  |
| `phase_03_eval_opt_verbosity` | `str` | `INFO` |  |
| `phase_04_registry_opt_description` | `str` | `` |  |
| `phase_04_registry_opt_format_name` | `str` | `pytorch` |  |
| `phase_04_registry_opt_format_version` | `str` | `1.0` |  |
| `phase_04_registry_opt_port` | `int` | `8080` |  |

## Metadata 🗂️

- **Name**: sft_pipeline
- **Tier**: core
- **Stability**: alpha
- **Dependencies**:
  - Kubeflow:
    - Name: Pipelines, Version: >=2.15.2
    - Name: Trainer, Version: >=0.1.0
  - External Services:
    - Name: HuggingFace Datasets, Version: >=2.14.0
    - Name: Kubernetes, Version: >=1.28.0
    - Name: Model Registry, Version: >=0.3.4
- **Tags**:
  - training
  - fine_tuning
  - sft
  - supervised_fine_tuning
  - llm
  - language_model
  - pipeline
- **Last Verified**: 2026-01-09 00:00:00+00:00
- **Owners**:
  - Approvers:
    - kramaranya
    - briangallagher
    - MStokluska
    - Fiona-Waters
    - szaher

## Additional Resources 📚

- **Documentation**: [https://github.com/kubeflow/trainer](https://github.com/kubeflow/trainer)
