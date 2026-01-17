# Osft Minimal Pipeline ✨

> ⚠️ **Stability: alpha** — This asset is not yet stable and may change.

## Overview 🧾

OSFT Minimal Training Pipeline - Continual learning without catastrophic forgetting.

A minimal 4-stage ML pipeline for fine-tuning language models with OSFT:

1) Dataset Download – Uses the shared ``dataset_download`` component to resolve datasets from Hugging Face,
   S3, HTTP, or PVC into validated chat-format JSONL train/eval splits on the workspace PVC and artifact store.
2) OSFT Training – Uses the shared ``train_model`` component with the TrainingHub mini-trainer backend to run
   Orthogonal Subspace Fine-Tuning with a reduced set of hyperparameters tuned for quick experimentation.
3) Evaluation – Uses the ``universal_llm_evaluator`` component to evaluate the minimal OSFT model on a small set
   of benchmarks (for example ARC-Easy) and optional custom data.
4) Model Registry – Uses the ``kubeflow_model_registry`` component to register the fine-tuned model in Kubeflow
   Model Registry when a registry address is provided.

For advanced configuration (more training and evaluation controls), see the full ``osft_pipeline``.

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
- **Prerequisites**: [- **Configured pipeline server (required)**
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
](- **Configured pipeline server (required)**
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
)

<!-- custom-content -->

## Compiling the Pipeline

This pipeline is defined as a Python function in `pipeline.py` and must be **compiled to a pipeline YAML** before it
can be imported into OpenShift AI.

1. **Ensure KFP SDK version**  
   Install or upgrade to **Kubeflow Pipelines SDK >= 2.15.2** in your environment:

   ```bash
   pip install "kfp>=2.15.2"
   ```

2. **Compile the pipeline**  
   From the repository root (or your chosen working directory), run a small Python snippet that imports and compiles
   the pipeline, for example:

   ```python
   from kfp import dsl
   from pipelines.training.osft_minimal import pipeline as osft_minimal_pipeline_module

   @dsl.pipeline
   def compiled_osft_minimal_pipeline(**kwargs):
       return osft_minimal_pipeline_module.osft_minimal_pipeline(**kwargs)

   if __name__ == "__main__":
       from kfp import compiler

       compiler.Compiler().compile(
           pipeline_func=compiled_osft_minimal_pipeline,
           package_path="osft_minimal_pipeline.yaml",
       )
   ```

   This will produce an `osft_minimal_pipeline.yaml` file that you can upload to OpenShift AI.

3. **Import the compiled pipeline into OpenShift AI**  
   Use the OpenShift AI console to import the compiled YAML as a new pipeline. For detailed UI steps, see the Red Hat
   documentation:  
   - [Importing a pipeline](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/3.0/html-single/working_with_ai_pipelines/index#importing-a-pipeline_ai-pipelines)

Once imported, you can create runs and experiments for this pipeline in the OpenShift AI UI.

## Managing Kubernetes Secrets for This Pipeline 🔐

This pipeline expects several Kubernetes secrets to be present in the **same namespace** as your pipeline runs. To
make setup easier, this directory includes a `secrets` subfolder with:

- **Secret manifests**:
  - `kubernetes-credentials.yaml`
  - `hf-token.yaml`
  - `s3-secret.yaml`
  - `oci-pull-secret-model-download.yaml`
- **Helper script**:
  - `create_secrets.sh`

Before using the script:

1. Open each YAML file under `secrets/` and **replace the dummy values** with real credentials for your environment.
2. Save the changes locally.

To create all four secrets in the **current kubectl namespace**:

```bash
cd pipelines-components/pipelines/training/osft_minimal/secrets
./create_secrets.sh
```

To target a specific namespace (for example, `my-project`):

```bash
./create_secrets.sh my-project
```

To create only a subset of secrets (for example, just `kubernetes-credentials` and `hf-token`):

```bash
./create_secrets.sh my-project kubernetes-credentials hf-token
```

If you prefer, you can **create the same secrets via the OpenShift AI / OpenShift UI** instead of using the script; the
YAML files serve as a reference for the expected keys and structure.
