# Osft Minimal Pipeline ✨

> ⚠️ **Stability: alpha** — This asset is not yet stable and may change.

## Overview 🧾

OSFT Minimal Training Pipeline - Continual learning without catastrophic forgetting.

A minimal 4-stage ML pipeline for fine-tuning language models with OSFT:

1) Dataset Download - Prepares training data from HuggingFace, S3, HTTP, or PVC
2) OSFT Training - Fine-tunes using mini-trainer backend (orthogonal subspace)
3) Evaluation - Evaluates with lm-eval harness (MMLU, GSM8K, etc.)
4) Model Registry - Registers trained model to Kubeflow Model Registry

## Quick start 🚀

For a minimal run of this pipeline:

- **Required parameter**:
  - Set `phase_01_dataset_man_data_uri` to a supported dataset, for example:
    - `hf://LipengCS/Table-GPT:All`
  - All other inputs have reasonable defaults and can be left unchanged for a first run.
- **Pipeline server**: Must be configured for your OpenShift AI project (see Prerequisites below).
- **Model Registry**: Optional. Leave `phase_04_registry_man_address` empty to skip registration.
- **Secrets**:
  - **Required**: `kubernetes-credentials` (provides `KUBERNETES_SERVER_URL` and `KUBERNETES_AUTH_TOKEN`) so the
    training step can submit a Training Hub OSFT job.
  - **Optional but often needed**:
    - `hf-token` (`HF_TOKEN`) if the chosen HF dataset or model is gated/private.
    - `s3-secret` if you use an `s3://` dataset URI instead of HF/HTTP/PVC.
    - `oci-pull-secret-model-download` if you use an `oci://` base model from a private registry.

## Prerequisites ✅

- **Configured pipeline server (required)**
  Before you can upload or run this pipeline, your OpenShift AI project must have a **pipeline server**
  configured. The pipeline server defines where pipeline **artifacts and run data are stored**
  (S3-compatible object storage) and is the endpoint that executes compiled pipeline YAML.
  Follow the Red Hat OpenShift AI documentation for configuring a pipeline server:
  - [Configuring a pipeline server in OpenShift AI](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/3.0/html/working_with_ai_pipelines/managing-ai-pipelines_ai-pipelines#configuring-a-pipeline-server_ai-pipelines)
  Without this, you will not be able to import the compiled YAML or create pipeline runs.

- **Model Registry (optional, but required if you want registration)**
  Stage 4 of this pipeline can **register the fine-tuned model** into a Model Registry instance.
  If you plan to use this step (that is, you set `phase_04_registry_man_address`), you must first create and
  configure a Model Registry in the **same project/namespace** as the pipeline server, and note its service
  address and port. See the Red Hat documentation for details:
  - [Creating a Model Registry](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/3.0/html/managing_model_registries/creating-a-model-registry_managing-model-registries)
  The pipeline’s Model Registry component uses this endpoint to register a new model version with training and
  evaluation metadata attached.

- **Kubernetes secrets (required and optional)**
  All secrets must be created in the **same namespace as the pipeline server and pipeline runs** (for example,
  your OpenShift AI project namespace). They are mounted into tasks as environment variables via
  `kfp.kubernetes.use_secret_as_env`.

  - **`s3-secret` (required for `s3://` datasets)**
    - **Used by**: Dataset download step when `dataset_uri` starts with `s3://`.
    - **Keys (in `data` or `stringData`)**:
      - `AWS_ACCESS_KEY_ID`
      - `AWS_SECRET_ACCESS_KEY`
    - These are injected as env vars into the dataset component. For `s3://` URIs, both must be set and
      non-empty; otherwise the component fails fast with a clear error.

  - **`hf-token` (optional, recommended for gated models/datasets)**
    - **Used by**: Dataset download, training, and evaluation steps.
    - **Key**:
      - `HF_TOKEN` – a valid Hugging Face access token.
    - The pipeline injects this into all main tasks as the `HF_TOKEN` environment variable. It is required for
      **gated/private** Hugging Face datasets or models; if absent, the pipeline can only access public,
      non-gated assets and will log a warning.

  - **`kubernetes-credentials` (required)**
    - **Used by**: Training step (`train_model` component) to talk to the Kubernetes API and submit Training Hub
      OSFT jobs.
    - **Keys**:
      - `KUBERNETES_SERVER_URL` – API server URL (for example, `https://api.<cluster-domain>:6443`).
      - `KUBERNETES_AUTH_TOKEN` – Bearer token with permission to list TrainingHub runtimes and manage jobs.
    - These are mounted as env vars and the training component **requires both to be set and non-empty**; if they
      are missing or mismatched, the component raises a clear configuration error instead of silently falling back.

  - **`oci-pull-secret-model-download` (optional, required for private OCI base models)**
    - **Used by**: Training step when `phase_02_train_man_train_model` is an `oci://` reference.
    - **Key**:
      - `OCI_PULL_SECRET_MODEL_DOWNLOAD` – the contents of a Docker `config.json`, used by `skopeo` to
        authenticate to the registry.
    - If this secret is not provided and the registry requires authentication, the OCI model download will fail
      with an authorization error. A common starting point is the cluster-wide pull secret; a cluster administrator
      can extract it with:

      oc get secret pull-secret -n openshift-config \
        -o jsonpath='{.data.\.dockerconfigjson}' | base64 -d | jq .

      You can then trim this `config.json` down to just the registries needed for your model and store it as the
      value of `OCI_PULL_SECRET_MODEL_DOWNLOAD` in an `Opaque` secret named `oci-pull-secret-model-download`.

## Inputs 📥

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `phase_01_dataset_man_data_uri` | `str` | `None` | [REQUIRED] Dataset location (hf://dataset, s3://bucket/path, https://url, pvc://path) |
| `phase_01_dataset_man_data_split` | `float` | `0.9` | Train/eval split ratio (0.9 = 90% train, 10% eval) |
| `phase_02_train_man_train_batch` | `int` | `128` | Effective batch size (samples per optimizer step) |
| `phase_02_train_man_train_epochs` | `int` | `1` | Number of training epochs. OSFT typically needs 1-2 |
| `phase_02_train_man_train_gpu` | `int` | `1` | GPUs per worker. OSFT handles multi-GPU well |
| `phase_02_train_man_train_model` | `str` | `Qwen/Qwen2.5-1.5B-Instruct` | Base model (HuggingFace ID or path) |
| `phase_02_train_man_train_tokens` | `int` | `64000` | Max tokens per GPU (memory cap). 64000 for OSFT |
| `phase_02_train_man_train_unfreeze` | `float` | `0.25` | [OSFT] Fraction to unfreeze (0.1=minimal, 0.25=balanced, 0.5=strong) |
| `phase_02_train_man_train_workers` | `int` | `1` | Number of training pods. OSFT efficient single-node (1) |
| `phase_03_eval_man_eval_tasks` | `list` | `['arc_easy']` | lm-eval tasks (arc_easy, mmlu, gsm8k, hellaswag, etc.) |
| `phase_04_registry_man_address` | `str` | `` | Model Registry address (empty = skip registration) |
| `phase_04_registry_man_reg_name` | `str` | `osft-model` | Model name in registry |
| `phase_04_registry_man_reg_version` | `str` | `1.0.0` | Semantic version (major.minor.patch) |
| `phase_01_dataset_opt_hf_token` | `str` | `` | HuggingFace token for gated/private datasets |
| `phase_01_dataset_opt_subset` | `int` | `0` | Limit to first N examples (0 = all) |
| `phase_02_train_opt_learning_rate` | `float` | `5e-06` | Learning rate (1e-6 to 1e-4). 5e-6 recommended |
| `phase_02_train_opt_max_seq_len` | `int` | `8192` | Max sequence length in tokens |
| `phase_02_train_opt_use_liger` | `bool` | `True` | [OSFT] Enable Liger kernel optimizations. Recommended |
| `phase_04_registry_opt_port` | `int` | `8080` | Model registry server port |

## Metadata 🗂️

- **Name**: osft_minimal_pipeline
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
  - osft
  - orthogonal_subspace
  - continual_learning
  - minimal
  - llm
  - language_model
  - pipeline
- **Last Verified**: 2026-01-14 00:00:00+00:00
- **Owners**:
  - Approvers:
    - briangallagher
    - Fiona-Waters
    - kramaranya
    - MStokluska
    - szaher

## Additional Resources 📚

- **Documentation**: [https://github.com/kubeflow/trainer](https://github.com/kubeflow/trainer)
