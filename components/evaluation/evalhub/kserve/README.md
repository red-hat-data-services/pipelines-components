# Evalhub Kserve ✨

> ⚠️ **Stability: experimental** — This asset is not yet stable and may change.

## Overview 🧾

Evaluate a model via Eval Hub with a KServe InferenceService.

Creates a KServe ServingRuntime + InferenceService (matching the RHOAI dashboard deployment pattern) to serve the fine-tuned model from the workspace PVC. The InferenceService URL is submitted to Eval Hub for benchmark evaluation. Both resources are cleaned up after completion.

## Inputs 📥

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `output_metrics` | `dsl.Output[dsl.Metrics]` | `None` | KFP Metrics artifact for evaluation scores. |
| `output_results` | `dsl.Output[dsl.Artifact]` | `None` | KFP Artifact for full evaluation results JSON. |
| `evalhub_url` | `str` | `None` | Eval Hub API endpoint (empty = skip evaluation). |
| `benchmarks` | `list` | `[]` | List of benchmark specs [{"provider_id": "...", "id": "..."}]. |
| `collection_id` | `str` | `""` | Eval Hub collection ID (alternative to benchmarks list). |
| `pvc_mount_path` | `str` | `""` | Workspace PVC mount path (triggers KFP PVC mount). |
| `model_artifact` | `dsl.Input[dsl.Model]` | `None` | Model artifact from upstream training step. |
| `model_path` | `str` | `None` | Local filesystem path to model directory (if no artifact). |
| `evalhub_tenant` | `str` | `""` | Eval Hub tenant / namespace header (X-Tenant). |
| `evalhub_auth_token` | `str` | `""` | Bearer token for Eval Hub auth. |
| `evalhub_model_name` | `str` | `finetuned-model` | Display name for the model in Eval Hub. |
| `base_model_name` | `str` | `""` | HF model ID for tokenizer resolution. |
| `evalhub_job_name` | `str` | `pipeline-eval` | Evaluation job name in Eval Hub. |
| `evalhub_timeout` | `int` | `7200` | Max seconds to wait for evaluation to complete. |
| `evalhub_poll_interval` | `int` | `30` | Seconds between eval status polls. |
| `mlflow_experiment_name` | `str` | `""` | MLflow experiment name (non-empty enables MLflow). |
| `gpu_count` | `int` | `1` | Number of GPUs for the InferenceService predictor. |
| `memory` | `str` | `8Gi` | Pod memory request/limit for the predictor (e.g. "8Gi", "32Gi"). |
| `cpu` | `str` | `2` | CPU request/limit for the predictor (e.g. "2"). |
| `runtime_image` | `str` | `registry.redhat.io/rhaii/vllm-cuda-rhel9@sha256:ad06abf3bb5235ebb5b2df84cd1b9fd09e823f0ff2eebfc82bb4590275ccfe0b` | Container image for the ServingRuntime (RHOAI vLLM default). |
| `trust_remote_code` | `bool` | `False` | Pass --trust-remote-code to vLLM (enables arbitrary code from model repos). |
| `verify_tls` | `bool` | `False` | Verify TLS certificates for Eval Hub API calls (False for self-signed certs). |
| `isvc_ready_timeout` | `int` | `600` | Max seconds to wait for InferenceService readiness. |

## Metadata 🗂️

- **Name**: evalhub_kserve
- **Stability**: experimental
- **Dependencies**:
  - Kubeflow:
    - Name: Pipelines, Version: >=2.15.2
  - External Services:
    - Name: Eval Hub, Version: >=0.1.0
    - Name: KServe, Version: >=0.11.0
    - Name: Kubernetes, Version: >=1.28.0
    - Name: vLLM (RHOAI), Version: >=0.6.0
- **Tags**:
  - evaluation
  - llm
  - eval_hub
  - kserve
  - benchmarks
  - metrics
  - mlflow
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
