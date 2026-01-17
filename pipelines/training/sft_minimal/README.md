# Sft Minimal Pipeline ✨

> ⚠️ **Stability: alpha** — This asset is not yet stable and may change.

## Overview 🧾

SFT Training Pipeline - Standard supervised fine-tuning with instructlab-training.

A minimal 4-stage ML pipeline for fine-tuning language models with SFT:

1) Dataset Download – Uses the shared ``dataset_download`` component to fetch and validate chat-format datasets
   from Hugging Face, S3, HTTP, or PVC, writing train/eval JSONL files to the workspace PVC.
2) SFT Training – Uses the shared ``train_model`` component with the TrainingHub ``instructlab-training``
   backend and a reduced set of hyperparameters suitable for quick trials.
3) Evaluation – Uses the ``universal_llm_evaluator`` component to run a small set of benchmarks (for example
   ARC-Easy) and optional custom holdout data against the fine-tuned model.
4) Model Registry – Uses the ``kubeflow_model_registry`` component to optionally register the fine-tuned model in
   Kubeflow Model Registry when a registry address is configured.

For more advanced SFT configuration (FSDP, logging options, additional knobs), see the full ``sft_pipeline``.

## Inputs 📥

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `phase_01_dataset_man_data_uri` | `str` | `None` | [REQUIRED] Dataset location (hf://dataset, s3://bucket/path, https://url, pvc://path) |
| `phase_01_dataset_man_data_split` | `float` | `0.9` | Train/eval split (0.9 = 90% train/10% eval, 1.0 = no split, all for training) |
| `phase_02_train_man_train_batch` | `int` | `128` | Effective batch size (samples per optimizer step). Start with 128 |
| `phase_02_train_man_epochs` | `int` | `1` | Number of training epochs. 1 is often sufficient |
| `phase_02_train_man_gpu` | `int` | `1` | GPUs per worker. KEEP AT 1 to avoid /dev/shm issues |
| `phase_02_train_man_model` | `str` | `Qwen/Qwen2.5-1.5B-Instruct` | Base model (HuggingFace ID or path) |
| `phase_02_train_man_tokens` | `int` | `10000` | Max tokens per GPU (memory cap). 10000 for SFT |
| `phase_02_train_man_workers` | `int` | `4` | Number of training pods. 4 pods × 1 GPU = 4 total GPUs |
| `phase_03_eval_man_tasks` | `list` | `['arc_easy']` | lm-eval tasks (arc_easy, mmlu, gsm8k, hellaswag, etc.) |
| `phase_04_registry_man_address` | `str` | `` | Model Registry address (empty = skip registration) |
| `phase_04_registry_man_reg_name` | `str` | `sft-model` | Model name in registry |
| `phase_04_registry_man_version` | `str` | `1.0.0` | Semantic version (major.minor.patch) |
| `phase_01_dataset_opt_subset` | `int` | `0` | Limit to first N examples (0 = all) |
| `phase_02_train_opt_env_vars` | `str` | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True, NCCL_DEBUG=INFO, NCCL_P2P_DISABLE=1, INSTRUCTLAB_NCCL_TIMEOUT_MS=60000` | Env vars (KEY=VAL,...) with NCCL timeout and memory optimization |
| `phase_02_train_opt_learning_rate` | `float` | `5e-06` | Learning rate (1e-6 to 1e-4). 5e-6 recommended |
| `phase_02_train_opt_max_seq_len` | `int` | `8192` | Max sequence length in tokens |
| `phase_02_train_opt_fsdp_sharding` | `str` | `FULL_SHARD` | FSDP strategy (FULL_SHARD, HYBRID_SHARD, NO_SHARD) |
| `phase_02_train_opt_use_liger` | `bool` | `False` | Enable Liger kernel optimizations |
| `phase_04_registry_opt_port` | `int` | `8080` | Model Registry server port. |

## Metadata 🗂️

- **Name**: sft_minimal_pipeline
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
  - sft
  - supervised_fine_tuning
  - minimal
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

   ```bash
   pip install "kfp>=2.15.2"
   ```

2. **Compile the pipeline**  
   From the repository root (or your chosen working directory), run a small Python snippet that imports and compiles
   the pipeline, for example:

   ```python
   from kfp import dsl
   from pipelines.training.sft_minimal import pipeline as sft_minimal_pipeline_module

   @dsl.pipeline
   def compiled_sft_minimal_pipeline(**kwargs):
       return sft_minimal_pipeline_module.sft_minimal_pipeline(**kwargs)

   if __name__ == "__main__":
       from kfp import compiler

       compiler.Compiler().compile(
           pipeline_func=compiled_sft_minimal_pipeline,
           package_path="sft_minimal_pipeline.yaml",
       )
   ```

   This will produce an `sft_minimal_pipeline.yaml` file that you can upload to OpenShift AI.

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
cd pipelines-components/pipelines/training/sft_minimal/secrets
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
