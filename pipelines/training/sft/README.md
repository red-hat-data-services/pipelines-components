# Sft Pipeline ✨

> ⚠️ **Stability: alpha** — This asset is not yet stable and may change.

## Overview 🧾

SFT Training Pipeline - Standard supervised fine-tuning with instructlab-training.

A 4-stage ML pipeline for fine-tuning language models:

1) Dataset Download - Prepares training data from HuggingFace, S3, HTTP, or PVC
2) SFT Training - Fine-tunes using instructlab-training backend
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
    training step can submit a Training Hub SFT job.
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
      SFT jobs.
    - **Keys**:
      - `KUBERNETES_SERVER_URL` – API server URL (for example, `https://api.<cluster-domain>:6443`).
      - `KUBERNETES_AUTH_TOKEN` – Bearer token with permission to list TrainingHub runtimes and manage jobs.
    - These are mounted as env vars and the training component **requires both to be set and non-empty**; if they
      are missing or mismatched, the component raises a clear configuration error instead of silently falling back.

  - **`oci-pull-secret-model-download` (optional, required for private OCI base models)**
    - **Used by**: Training step when `phase_02_train_man_model` is an `oci://` reference.
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
| `phase_01_dataset_man_data_uri` | `str` | `None` | Dataset location (hf://, s3://, https://, pvc://). |
| `phase_01_dataset_man_data_split` | `float` | `0.9` | Train/eval split ratio (0.9 = 90% train). |
| `phase_02_train_man_train_batch` | `int` | `128` | Effective batch size per optimizer step. |
| `phase_02_train_man_epochs` | `int` | `1` | Number of training epochs. |
| `phase_02_train_man_gpu` | `int` | `1` | GPUs per worker. Keep at 1 to avoid /dev/shm issues. |
| `phase_02_train_man_model` | `str` | `Qwen/Qwen2.5-1.5B-Instruct` | Base model (HuggingFace ID or path). |
| `phase_02_train_man_tokens` | `int` | `10000` | Max tokens per GPU (memory cap). |
| `phase_02_train_man_workers` | `int` | `4` | Number of training pods. |
| `phase_03_eval_man_tasks` | `list` | `['arc_easy']` | lm-eval tasks (arc_easy, mmlu, gsm8k, etc.). |
| `phase_04_registry_man_address` | `str` | `` | Model Registry address (empty = skip). |
| `phase_04_registry_man_author` | `str` | `pipeline` | Author name for the registered model. |
| `phase_04_registry_man_name` | `str` | `sft-model` | Model name in registry. |
| `phase_04_registry_man_version` | `str` | `1.0.0` | Semantic version (major.minor.patch). |
| `phase_01_dataset_opt_hf_token` | `str` | `` | HuggingFace token for private datasets. |
| `phase_01_dataset_opt_subset` | `int` | `0` | Limit dataset to N samples (0 = all). |
| `phase_02_train_opt_annotations` | `str` | `` | Pod annotations as key=value,key=value. |
| `phase_02_train_opt_cpu` | `str` | `4` | CPU cores per worker. |
| `phase_02_train_opt_env_vars` | `str` | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,NCCL_DEBUG=INFO,INSTRUCTLAB_NCCL_TIMEOUT_MS=600000` | Environment variables as KEY=VAL,KEY=VAL. |
| `phase_02_train_opt_hf_token` | `str` | `` | HuggingFace token for gated models. |
| `phase_02_train_opt_labels` | `str` | `` | Pod labels as key=value,key=value. |
| `phase_02_train_opt_learning_rate` | `float` | `5e-06` | Learning rate for training. |
| `phase_02_train_opt_lr_warmup` | `int` | `0` | Learning rate warmup steps. |
| `phase_02_train_opt_lr_scheduler` | `str` | `cosine` | LR scheduler type (cosine, linear). |
| `phase_02_train_opt_max_seq_len` | `int` | `8192` | Maximum sequence length in tokens. |
| `phase_02_train_opt_memory` | `str` | `64Gi` | Memory per worker (e.g., 64Gi). |
| `phase_02_train_opt_num_procs` | `str` | `auto` | Processes per worker (auto or int). |
| `phase_02_train_opt_pull_secret` | `str` | `` | Pull secret for container registry. |
| `phase_02_train_opt_save_epoch` | `bool` | `True` | Save checkpoint at each epoch. |
| `phase_02_train_opt_save_full_state` | `bool` | `False` | Save full accelerate state at epoch. |
| `phase_02_train_opt_fsdp_sharding` | `str` | `FULL_SHARD` | FSDP strategy (FULL_SHARD, HYBRID_SHARD). |
| `phase_02_train_opt_save_samples` | `int` | `0` | Number of samples to save (0 = none). |
| `phase_02_train_opt_seed` | `int` | `42` | Random seed for reproducibility. |
| `phase_02_train_opt_use_liger` | `bool` | `False` | Enable Liger kernel optimizations. |
| `phase_03_eval_opt_batch` | `str` | `auto` | Batch size for evaluation (auto or int). |
| `phase_03_eval_opt_gen_kwargs` | `dict` | `{}` | Generation kwargs for evaluation. |
| `phase_03_eval_opt_limit` | `int` | `-1` | Limit examples per task (-1 = no limit). |
| `phase_03_eval_opt_log_samples` | `bool` | `True` | Log individual evaluation samples. |
| `phase_03_eval_opt_model_args` | `dict` | `{}` | Model initialization arguments. |
| `phase_03_eval_opt_verbosity` | `str` | `INFO` | Logging verbosity (DEBUG, INFO, etc.). |
| `phase_04_registry_opt_description` | `str` | `` | Model description for registry. |
| `phase_04_registry_opt_format_name` | `str` | `pytorch` | Model format (pytorch, onnx). |
| `phase_04_registry_opt_format_version` | `str` | `1.0` | Model format version. |
| `phase_04_registry_opt_port` | `int` | `8080` | Model Registry server port. |

## Metadata 🗂️

- **Name**: sft_pipeline
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
    - briangallagher
    - Fiona-Waters
    - kramaranya
    - MStokluska
    - szaher

## Troubleshooting 🛠️

- **Kubernetes credentials / TrainingHub errors**
  - Symptom: Training step fails with errors like `Failed to list ClusterTrainingRuntimes` or HTTP `401 Unauthorized`.
  - Likely cause: `kubernetes-credentials` secret missing or misconfigured:
    - Ensure the secret exists in the pipeline namespace.
    - Ensure it has **both** `KUBERNETES_SERVER_URL` and `KUBERNETES_AUTH_TOKEN` keys set and non-empty.
  - The training component requires both env vars and will raise a clear configuration error if they are missing.

- **S3 dataset failures**
  - Symptom: Dataset step fails with a `ValueError` mentioning `S3 credentials misconfigured` or
    `S3 credentials missing`.
  - Likely cause: `s3-secret` secret missing or only one of `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` is set.
  - Fix: Create/update `s3-secret` in the pipeline namespace with both keys populated; for `s3://` URIs both must
    be set.

- **OCI model download failures**
  - Symptom: Training step logs show `skopeo copy failed` with `unauthorized` or `authentication required`.
  - Likely cause: `oci-pull-secret-model-download` secret is missing or does not contain a valid Docker
    `config.json` as `OCI_PULL_SECRET_MODEL_DOWNLOAD`.
  - Fix: Create the secret in the pipeline namespace with a trimmed `config.json` that includes credentials for
    the registries hosting your model image (see Prerequisites above for an example command to extract the
    cluster-wide pull secret as a starting point).

- **Hugging Face gated datasets or models**
  - Symptom: Dataset or training step fails with 403/permission errors from Hugging Face, or logs warnings that
    `HF_TOKEN` is not set when using HF IDs (for example, gated models or datasets).
  - Likely cause: `hf-token` secret is missing or the token does not have access to the requested asset.
  - Fix: Create `hf-token` with a valid `HF_TOKEN` value in the pipeline namespace, and ensure the token has
    accepted the dataset/model license terms on the Hugging Face Hub.

- **Chat-format validation errors**
  - Symptom: Dataset step fails with messages like “missing 'messages' or 'conversations' field” or invalid `role`.
  - Likely cause: Input data is not in the expected chat format (a list of messages/conversations with `role` and
    `content` fields, roles in `{system, user, assistant, function, tool}`).
  - Fix: Inspect the source dataset (JSON/JSONL or HF dataset) and adjust it to match the expected structure, or
    choose a dataset that already uses a compatible chat schema.

## Additional Resources 📚

- **Documentation**: [https://github.com/kubeflow/trainer](https://github.com/kubeflow/trainer)
