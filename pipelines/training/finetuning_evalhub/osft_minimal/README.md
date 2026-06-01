# Osft Minimal Evalhub Pipeline ✨

> ⚠️ **Stability: experimental** — This asset is not yet stable and may change.

## Overview 🧾

OSFT Training Pipeline — Eval Hub Minimal.

Minimal-config variant that runs 5 leaderboard benchmarks (ifeval, bbh, mmlu_pro, musr, math_hard) via KServe model serving — all public datasets, no HF token required, no trust_remote_code issues.

Prerequisites: Eval Hub and KServe must be installed on the cluster. The pipeline ServiceAccount needs RBAC permissions for inferenceservices.serving.kserve.io and servingruntimes.serving.kserve.io resources (create, delete, get, list, patch). The workspace PVC must use ReadWriteMany access mode
(NFS-backed) so the KServe predictor pod can mount the model. The eval component uses the in-cluster ServiceAccount token for K8s API access.

## Inputs 📥

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `phase_01_dataset_man_data_uri` | `str` | `None` | Dataset location (hf://, s3://, https://). |
| `phase_02_train_man_train_batch` | `int` | `128` | Effective batch size (samples per optimizer step). |
| `phase_02_train_man_train_epochs` | `int` | `1` | Number of training epochs. |
| `phase_02_train_man_train_gpu` | `int` | `1` | GPUs per worker. |
| `phase_02_train_man_train_model` | `str` | `Qwen/Qwen2.5-1.5B-Instruct` | Base model (HuggingFace ID or path). |
| `phase_02_train_man_train_tokens` | `int` | `64000` | Max tokens per GPU (memory cap). |
| `phase_02_train_man_train_unfreeze` | `float` | `0.25` | Fraction to unfreeze (0.1=minimal, 0.25=balanced, 0.5=strong). |
| `phase_02_train_man_train_workers` | `int` | `1` | Number of training pods. |
| `phase_03_eval_opt_evalhub_url` | `str` | `""` | Eval Hub API endpoint URL (empty = skip evaluation). |
| `phase_03_eval_opt_mlflow_experiment` | `str` | `""` | MLflow experiment name (non-empty = enable, empty = disabled). |
| `phase_03_eval_opt_kserve_gpu_count` | `int` | `1` | GPUs for the KServe predictor. |
| `phase_03_eval_opt_kserve_cpu` | `str` | `2` | CPU for the KServe predictor. |
| `phase_03_eval_opt_kserve_memory` | `str` | `32Gi` | Pod memory for the KServe predictor. |
| `phase_03_eval_opt_timeout` | `int` | `7200` | Max seconds to wait for evaluation. |
| `phase_04_registry_man_address` | `str` | `""` | Model Registry address (empty = skip). |
| `phase_04_registry_man_reg_author` | `str` | `pipeline` | Author name for the registered model. |
| `phase_04_registry_man_reg_name` | `str` | `osft-model` | Model name in registry. |
| `phase_04_registry_man_reg_version` | `str` | `1.0.0` | Semantic version. |
| `phase_01_dataset_opt_subset` | `int` | `0` | Limit to first N examples (0 = all). |
| `phase_02_train_opt_cpu` | `str` | `8` | CPU cores per worker. |
| `phase_02_train_opt_env_vars` | `str` | `""` | Env vars (KEY=VAL,...). |
| `phase_02_train_opt_learning_rate` | `float` | `5e-06` | Learning rate. 5e-6 for OSFT. |
| `phase_02_train_opt_max_seq_len` | `int` | `8192` | Max sequence length in tokens. |
| `phase_02_train_opt_memory` | `str` | `32Gi` | RAM per worker. |
| `phase_02_train_opt_use_liger` | `bool` | `True` | Enable Liger kernel optimizations. |
| `phase_02_train_opt_runtime` | `str` | `training-hub` | ClusterTrainingRuntime name. |
| `phase_04_registry_opt_port` | `int` | `8080` | Model registry server port. |

## Metadata 🗂️

- **Name**: osft_minimal_evalhub_pipeline
- **Stability**: experimental
- **Dependencies**:
  - Kubeflow:
    - Name: Pipelines, Version: >=2.15.2
    - Name: Trainer, Version: >=0.1.0
  - External Services:
    - Name: Eval Hub, Version: >=0.1.0
    - Name: KServe, Version: >=0.11.0
    - Name: Kubernetes, Version: >=1.28.0
- **Tags**:
  - training
  - fine_tuning
  - osft
  - orthogonal_subspace_fine_tuning
  - eval_hub
  - kserve
  - minimal
  - pipeline
- **Last Verified**: 2026-05-20 00:00:00+00:00
- **Owners**:
  - Approvers:
    - briangallagher
    - Fiona-Waters
    - kramaranya
    - MStokluska
    - szaher

## Additional Resources 📚

- **Documentation**: [https://github.com/opendatahub-io/eval-hub](https://github.com/opendatahub-io/eval-hub)
