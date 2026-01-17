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

## Additional Resources 📚

- **Documentation**: [https://github.com/kubeflow/trainer](https://github.com/kubeflow/trainer)

<!-- custom-content -->

## Compiling the Pipeline

This pipeline is defined as a Python function in `pipeline.py` and must be **compiled to a pipeline YAML** before it
can be imported into OpenShift AI.

1. **Ensure KFP SDK version**  
   Install or upgrade to **Kubeflow Pipelines SDK >= 2.15.2** in your environment:

       pip install "kfp>=2.15.2"

2. **Compile the pipeline**  
   From the repository root (or your chosen working directory), run a small Python snippet that imports and compiles
   the pipeline, for example:

       from kfp import dsl
       from pipelines.training.osft import pipeline as osft_pipeline_module

       @dsl.pipeline
       def compiled_osft_pipeline(**kwargs):
           return osft_pipeline_module.osft_pipeline(**kwargs)

       if __name__ == "__main__":
           from kfp import compiler

           compiler.Compiler().compile(
               pipeline_func=compiled_osft_pipeline,
               package_path="osft_pipeline.yaml",
           )

   This will produce an `osft_pipeline.yaml` file that you can upload to OpenShift AI.

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

    cd pipelines-components/pipelines/training/osft/secrets
    ./create_secrets.sh

To target a specific namespace (for example, `my-project`):

    ./create_secrets.sh my-project

To create only a subset of secrets (for example, just `kubernetes-credentials` and `hf-token`):

    ./create_secrets.sh my-project kubernetes-credentials hf-token

If you prefer, you can **create the same secrets via the OpenShift AI / OpenShift UI** instead of using the script; the
YAML files serve as a reference for the expected keys and structure.
