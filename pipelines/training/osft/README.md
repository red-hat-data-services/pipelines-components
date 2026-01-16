# Osft Pipeline ✨

> ⚠️ **Stability: alpha** — This asset is not yet stable and may change.

## Overview 🧾

OSFT Training Pipeline - Continual learning without catastrophic forgetting.

A 4-stage ML pipeline for fine-tuning language models with OSFT:

    1) Dataset Download – Resolves datasets from HuggingFace, S3, HTTP, or PVC, validates chat-format structure,
       and writes train/eval JSONL artifacts to the pipeline workspace and artifact store.
    2) OSFT Training – Uses the shared ``train_model`` component to submit a Training Hub job for Orthogonal
       Subspace Fine-Tuning via the mini-trainer backend, producing fine-tuned checkpoints and training metrics.
       For details on the underlying algorithms and backends, see the Training Hub project:
       https://github.com/Red-Hat-AI-Innovation-Team/training_hub
    3) Evaluation – Runs the ``universal_llm_evaluator`` component, which wraps the lm-evaluation-harness with a
       vLLM backend to compute benchmark scores (for example, MMLU, GSM8K) and optional custom holdout metrics.
    4) Model Registry – Uses the ``kubeflow_model_registry`` component to register the fine-tuned model, attaching
       training and evaluation metadata to a new or existing entry in Kubeflow Model Registry.

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
    training step can submit a Training Hub job.
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
    - **Used by**: Training step (`train_model` component) to talk to the Kubernetes API and submit TrainingHub
      jobs.
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

## Parameter naming convention 🧩

Pipeline input parameters follow a consistent naming pattern that encodes where and how they are used:

- **`phase_XX`**: The **pipeline stage number**, from `phase_01` (dataset) through `phase_04` (registry).
- **`<stage_name>`**: A short **stage identifier**, such as `dataset`, `train`, `eval`, or `registry`.
- **`man` / `opt`**: Whether the parameter is **mandatory** (`man`) or **optional** (`opt`) for typical runs.
- **`<param_name>`**: The actual **parameter meaning**, for example `data_uri`, `train_batch`, or `eval_tasks`.

For example:

- `phase_01_dataset_man_data_uri` → stage 1 (dataset), **mandatory**, dataset URI.
- `phase_02_train_opt_learning_rate` → stage 2 (training), **optional**, learning rate setting.

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
| `phase_04_registry_man_reg_author` | `str` | `pipeline` | Author name for the registered model |
| `phase_04_registry_man_reg_name` | `str` | `osft-model` | Model name in registry |
| `phase_04_registry_man_reg_version` | `str` | `1.0.0` | Semantic version (major.minor.patch) |
| `phase_01_dataset_opt_hf_token` | `str` | `` | HuggingFace token for gated/private datasets |
| `phase_01_dataset_opt_subset` | `int` | `0` | Limit to first N examples (0 = all) |
| `phase_02_train_opt_annotations` | `str` | `` | K8s annotations (key=val,...) |
| `phase_02_train_opt_cpu` | `str` | `8` | CPU cores per worker. 8 recommended for OSFT |
| `phase_02_train_opt_env_vars` | `str` | `` | Env vars (KEY=VAL,...). OSFT typically doesn't need special vars |
| `phase_02_train_opt_hf_token` | `str` | `` | HuggingFace token for gated models (Llama, Mistral) |
| `phase_02_train_opt_labels` | `str` | `` | K8s labels (key=val,...) |
| `phase_02_train_opt_learning_rate` | `float` | `5e-06` | Learning rate (1e-6 to 1e-4). 5e-6 recommended |
| `phase_02_train_opt_lr_scheduler` | `str` | `cosine` | [OSFT] LR schedule (cosine, linear, constant) |
| `phase_02_train_opt_lr_scheduler_kwargs` | `str` | `` | [OSFT] Extra scheduler params (key=val,...) |
| `phase_02_train_opt_lr_warmup` | `int` | `0` | Warmup steps before full LR |
| `phase_02_train_opt_max_seq_len` | `int` | `8192` | Max sequence length in tokens |
| `phase_02_train_opt_memory` | `str` | `32Gi` | RAM per worker. 32Gi usually sufficient for OSFT |
| `phase_02_train_opt_num_procs` | `str` | `auto` | Processes per worker ('auto' = one per GPU) |
| `phase_02_train_opt_processed_data` | `bool` | `False` | [OSFT] True if dataset already has tokenized input_ids |
| `phase_02_train_opt_pull_secret` | `str` | `` | K8s pull secret for private registries |
| `phase_02_train_opt_save_epoch` | `bool` | `False` | Save checkpoint at each epoch. Usually False for OSFT |
| `phase_02_train_opt_save_final` | `bool` | `True` | [OSFT] Save final checkpoint after all epochs |
| `phase_02_train_opt_seed` | `int` | `42` | Random seed for reproducibility |
| `phase_02_train_opt_target_patterns` | `str` | `` | [OSFT] Module patterns to unfreeze (empty=auto) |
| `phase_02_train_opt_unmask` | `bool` | `False` | [OSFT] Unmask all tokens (False=assistant only) |
| `phase_02_train_opt_use_liger` | `bool` | `True` | [OSFT] Enable Liger kernel optimizations. Recommended |
| `phase_03_eval_opt_batch` | `str` | `auto` | Eval batch size ('auto' or integer) |
| `phase_03_eval_opt_gen_kwargs` | `dict` | `{}` | Generation params dict (max_tokens, temperature) |
| `phase_03_eval_opt_limit` | `int` | `-1` | Max samples per task (-1 = all) |
| `phase_03_eval_opt_log_samples` | `bool` | `True` | Log individual predictions |
| `phase_03_eval_opt_model_args` | `dict` | `{}` | Model init args dict (dtype, gpu_memory_utilization) |
| `phase_03_eval_opt_verbosity` | `str` | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR) |
| `phase_04_registry_opt_description` | `str` | `` | Model description |
| `phase_04_registry_opt_format_name` | `str` | `pytorch` | Model format (pytorch, onnx, tensorflow) |
| `phase_04_registry_opt_format_version` | `str` | `1.0` | Model format version |
| `phase_04_registry_opt_port` | `int` | `8080` | Model registry server port |

## Outputs 📤

This pipeline produces both **metrics artifacts** (used for tracking performance and feeding the Model Registry)
and **data/model artifacts** (datasets, models, evaluation results). All artifacts are stored in the pipeline
server’s S3-compatible bucket under the pipeline run, as described in the OpenShift AI docs on pipeline storage
([Managing AI pipelines](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/3.0/html/working_with_ai_pipelines/managing-ai-pipelines_ai-pipelines#configuring-a-pipeline-server_ai-pipelines)).

- **Stage 1 – Dataset Download (`dataset_download`)**
  - **Data artifacts**:
    - `train_dataset` (`dsl.Dataset`): JSONL training data written to the artifact store and to the workspace PVC
      (path recorded in `train_dataset.metadata["pvc_path"]`, typically under `/kfp-workspace/datasets/train.jsonl`).
    - `eval_dataset` (`dsl.Dataset`): JSONL evaluation data, also stored as a KFP artifact and on the PVC
      (path in `eval_dataset.metadata["pvc_path"]`).
  - **Metrics**: None for this stage.

- **Stage 2 – Training (`train_model`)**
  - **Data/model artifacts**:
    - `output_model` (`dsl.Model`): The final checkpoint directory for the fine-tuned model, stored as a Model
      artifact in the artifact store. The component also copies this directory to the workspace PVC (path recorded
      in `output_model.metadata["pvc_model_dir"]`, typically `/kfp-workspace/final_model`).
  - **Metrics artifacts**:
    - `output_metrics` (`dsl.Metrics`): Contains hyperparameters (batch size, learning rate, etc.) and aggregated
      training metrics (loss, perplexity, etc.). These values are visible in the run UI and are consumed by the
      Model Registry step via `input_metrics`.

- **Stage 3 – Evaluation (`universal_llm_evaluator`)**
  - **Metrics artifacts**:
    - `output_metrics` (`dsl.Metrics`): Benchmark scores (for example `arc_easy_acc,none`) and summary fields
      such as `eval_duration_seconds`, `eval_tasks_count`, and custom holdout metrics when used. The Model
      Registry step reads these via `eval_metrics.metadata`.
  - **Result artifacts**:
    - `output_results` (`dsl.Artifact`): A JSON file (`eval_results.json`) containing the full lm-eval results
      structure, stored in the artifact store at the `output_results` URI.
    - `output_samples` (`dsl.Artifact`, optional): When sample logging is enabled, a JSON file
      (`eval_samples.json`) with prompt/response pairs for custom holdout or benchmark tasks.

- **Stage 4 – Model Registry (`kubeflow_model_registry`)**
  - **Model Registry state**:
    - Registers or updates a model version in the configured Model Registry instance, using the model URI and
      metadata from training and evaluation. The registered **model ID** is written to logs and returned as the
      component output.
  - **Logs on PVC**:
    - The component appends a summary line (including the model ID) to the shared log file on the workspace PVC
      (for example, `/kfp-workspace/pipeline_log.txt`), alongside messages from other stages.

## Metadata 🗂️

- **Name**: osft_pipeline
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
  - osft
  - orthogonal_subspace_fine_tuning
  - pipeline
- **Last Verified**: 2026-01-14 00:00:00+00:00
- **Owners**:
  - Approvers:
    - briangallagher
    - Fiona-Waters
    - kramaranya
    - MStokluska
    - szaher

## Customizing this pipeline 🔧

This pipeline is built from reusable components that live elsewhere in the repository. You can use them as-is, or
fork and compose your own pipelines:

- **Dataset download component**
  - Path: `components/data_processing/dataset_download/component.py`
  - Provides a generic dataset loader that understands HF, S3, HTTP, and PVC URIs, validates chat-format data,
    and writes train/eval JSONL outputs. See that component’s README for details on formats, validation rules,
    and configuration options.

- **Training component (OSFT/SFT)**
  - Path: `components/training/finetuning/component.py`
  - Wraps Training Hub’s OSFT/SFT algorithms and backends (mini-trainer, instructlab-training), handling
    Kubernetes integration, PVC-based caching, environment setup, and Training Hub job submission. Refer to its
    README for supported parameters, resource configuration, and how it maps to Training Hub options.

- **Evaluation component**
  - Path: `components/evaluation/lm_eval/component.py` (`universal_llm_evaluator`)
  - Runs lm-evaluation-harness with a vLLM backend over standard benchmarks and custom holdout datasets.
    Its README describes supported tasks, model arguments, and how metrics are logged and structured.

- **Model Registry component**
  - Path: `components/deployment/kubeflow_model_registry/component.py`
  - Registers models and versions in Kubeflow Model Registry, attaching training/eval metadata and provenance
    information. See its README for the mapping between inputs, metadata fields, and the resulting registry
    entries.

All of these components can be imported and used independently in other pipelines. To customize this OSFT
pipeline, you can:

- **Fork the pipeline definition** under `pipelines/training/osft` and adjust parameter defaults, wiring, or
  stages (for example, swap in a different evaluator or registry step).
- **Reuse individual components** in your own pipelines by referencing their Python entrypoints and following
  the usage documented in each component’s README.

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
