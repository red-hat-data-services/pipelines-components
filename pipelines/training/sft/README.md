# Sft Pipeline ✨

> ⚠️ **Stability: alpha** — This asset is not yet stable and may change.

## Overview 🧾

SFT Training Pipeline - Standard supervised fine-tuning with instructlab-training.

A 4-stage ML pipeline for fine-tuning language models:

1) Dataset Download – Uses the shared ``dataset_download`` component to pull datasets from Hugging Face, S3,
   HTTP, or PVC, validate chat-format structure, and materialize train/eval JSONL splits on the workspace PVC
   and in the artifact store.
2) SFT Training – Uses the shared ``train_model`` component with the TrainingHub ``instructlab-training``
   backend to run supervised fine-tuning, producing a fine-tuned checkpoint and training metrics for the base
   model.
3) Evaluation – Runs the ``universal_llm_evaluator`` component, which wraps lm-evaluation-harness (typically
   with a vLLM backend) to score the SFT model on benchmarks such as ARC-Easy, MMLU, and GSM8K, as well as
   optional custom holdout sets.
4) Model Registry – Uses the ``kubeflow_model_registry`` component to register the fine-tuned model in Kubeflow
   Model Registry, attaching both training and evaluation metadata to the model version.

## Inputs 📥

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `phase_01_dataset_man_data_uri` | `str` | `None` | Dataset location (hf://, s3://, https://, pvc://). |
| `phase_01_dataset_man_data_split` | `float` | `0.9` | Train/eval split (0.9 = 90% train/10% eval, 1.0 = no split, all for training). |
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
| `phase_01_dataset_opt_subset` | `int` | `0` | Limit dataset to N samples (0 = all). |
| `phase_02_train_opt_annotations` | `str` | `` | Pod annotations as key=value,key=value. |
| `phase_02_train_opt_cpu` | `str` | `4` | CPU cores per worker. |
| `phase_02_train_opt_env_vars` | `str` | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,NCCL_DEBUG=INFO,INSTRUCTLAB_NCCL_TIMEOUT_MS=600000` | Environment variables as KEY=VAL,KEY=VAL. |
| `phase_02_train_opt_labels` | `str` | `` | Pod labels as key=value,key=value. |
| `phase_02_train_opt_learning_rate` | `float` | `5e-06` | Learning rate for training. |
| `phase_02_train_opt_lr_warmup` | `int` | `0` | Learning rate warmup steps. |
| `phase_02_train_opt_lr_scheduler` | `str` | `cosine` | LR scheduler type (cosine, linear). |
| `phase_02_train_opt_max_seq_len` | `int` | `8192` | Maximum sequence length in tokens. |
| `phase_02_train_opt_memory` | `str` | `64Gi` | Memory per worker (e.g., 64Gi). |
| `phase_02_train_opt_num_procs` | `str` | `auto` | Processes per worker (auto or int). |
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

## Additional Resources 📚

- **Documentation**: [https://github.com/kubeflow/trainer](https://github.com/kubeflow/trainer)

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
   from pipelines.training.sft import pipeline as sft_pipeline_module

   @dsl.pipeline
   def compiled_sft_pipeline(**kwargs):
       return sft_pipeline_module.sft_pipeline(**kwargs)

   if __name__ == "__main__":
       from kfp import compiler

       compiler.Compiler().compile(
           pipeline_func=compiled_sft_pipeline,
           package_path="sft_pipeline.yaml",
       )
   ```

   This will produce an `sft_pipeline.yaml` file that you can upload to OpenShift AI.

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

1. Open each YAML file under `secrets/` and **replace the placeholder values** with real credentials for your environment.
2. Save the changes locally.

To create all four secrets in the **current kubectl namespace**:

```bash
cd pipelines-components/pipelines/training/sft/secrets
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
