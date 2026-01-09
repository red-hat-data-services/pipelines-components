# Finetuning ✨

## Overview 🧾

Train model using TrainingHub (OSFT/SFT). Outputs model artifact and metrics.

## Inputs 📥

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pvc_path` | `str` | `None` |  |
| `output_model` | `dsl.Output[dsl.Model]` | `None` |  |
| `output_metrics` | `dsl.Output[dsl.Metrics]` | `None` |  |
| `dataset` | `dsl.Input[dsl.Dataset]` | `None` |  |
| `training_base_model` | `str` | `Qwen/Qwen2.5-1.5B-Instruct` |  |
| `training_algorithm` | `str` | `OSFT` |  |
| `training_effective_batch_size` | `int` | `128` |  |
| `training_max_tokens_per_gpu` | `int` | `64000` |  |
| `training_max_seq_len` | `int` | `8192` |  |
| `training_learning_rate` | `Optional[float]` | `None` |  |
| `training_backend` | `str` | `mini-trainer` |  |
| `training_lr_warmup_steps` | `Optional[int]` | `None` |  |
| `training_checkpoint_at_epoch` | `Optional[bool]` | `None` |  |
| `training_num_epochs` | `Optional[int]` | `None` |  |
| `training_data_output_dir` | `Optional[str]` | `None` |  |
| `training_hf_token` | `str` | `` |  |
| `training_pull_secret` | `str` | `` |  |
| `training_envs` | `str` | `` |  |
| `training_resource_cpu_per_worker` | `str` | `8` |  |
| `training_resource_gpu_per_worker` | `int` | `1` |  |
| `training_resource_memory_per_worker` | `str` | `32Gi` |  |
| `training_resource_num_procs_per_worker` | `str` | `auto` |  |
| `training_resource_num_workers` | `int` | `1` |  |
| `training_metadata_labels` | `str` | `` |  |
| `training_metadata_annotations` | `str` | `` |  |
| `training_unfreeze_rank_ratio` | `float` | `0.25` |  |
| `training_osft_memory_efficient_init` | `bool` | `True` |  |
| `training_target_patterns` | `str` | `` |  |
| `training_seed` | `Optional[int]` | `None` |  |
| `training_use_liger` | `Optional[bool]` | `None` |  |
| `training_use_processed_dataset` | `Optional[bool]` | `None` |  |
| `training_unmask_messages` | `Optional[bool]` | `None` |  |
| `training_lr_scheduler` | `Optional[str]` | `None` |  |
| `training_lr_scheduler_kwargs` | `str` | `` |  |
| `training_save_final_checkpoint` | `Optional[bool]` | `None` |  |
| `training_save_samples` | `Optional[int]` | `None` |  |
| `training_accelerate_full_state_at_epoch` | `Optional[bool]` | `None` |  |
| `training_fsdp_sharding_strategy` | `Optional[str]` | `None` |  |
| `kubernetes_config` | `dsl.TaskConfig` | `None` |  |

## Outputs 📤

| Name | Type | Description |
|------|------|-------------|
| Output | `str` |  |

## Metadata 🗂️

- **Name**: finetuning
- **Tier**: core
- **Stability**: alpha
- **Dependencies**:
  - Kubeflow:
    - Name: Pipelines, Version: >=2.15.2
    - Name: Trainer, Version: >=0.1.0
  - External Services:
    - Name: HuggingFace Datasets, Version: >=2.14.0
    - Name: Kubernetes, Version: >=1.28.0
- **Tags**:
  - training
  - fine_tuning
  - finetuning
  - osft
  - sft
  - llm
  - language_model
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
