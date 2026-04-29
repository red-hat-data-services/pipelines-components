# Download Model Component

> **Stability: experimental** — This asset is not yet stable and may change.

KFP component that downloads a HuggingFace model to a PVC for caching. On subsequent runs, detects the model is already present and skips the download, avoiding re-downloading 14GB+ model weights every time.

## How It Works

1. **Derive Path** -- Converts the HuggingFace model ID to a directory-safe name by replacing `/` with `--` (e.g. `mistralai/Mistral-7B-Instruct-v0.3` becomes `mistralai--Mistral-7B-Instruct-v0.3`).
2. **Check Cache** -- Looks for `config.json` in the target directory on the PVC. If found, the download is skipped.
3. **Download** -- If not cached, calls `huggingface_hub.snapshot_download()` to download the full model snapshot (weights, tokenizer, config) to the PVC path without symlinks.
4. **Return Path** -- Returns the PVC sub-directory name (not the full path), which the downstream `model_deployment` component uses to construct the `storageUri`.

## Inputs

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name` | `str` | *required* | HuggingFace model ID (e.g. `mistralai/Mistral-7B-Instruct-v0.3`) |
| `model_cache_pvc` | `str` | *required* | Name of the PVC to store models (mounted via pipeline) |
| `model_cache_mount` | `str` | `"/mnt/models"` | Mount path for the model cache PVC |

## Output

| Type | Description |
|------|-------------|
| `str` | PVC sub-path where the model is stored (e.g. `mistralai--Mistral-7B-Instruct-v0.3`) |

This output is consumed by the `model_deployment` component to construct:
```
pvc://{model_cache_pvc}/{model_dir}
```

## Metadata

- **Name**: download_model
- **Stability**: experimental
- **Dependencies**:
  - Kubeflow:
    - Name: Pipelines, Version: >=2.15.2
  - External Services:
    - Name: HuggingFace Hub, Version: >=0.20.0
- **Tags**: data_processing, model_download, huggingface, caching, rag
- **Last Verified**: 2026-04-28
- **Owners**:
  - Approvers: szaher

## Dependencies

- **Base image**: `registry.redhat.io/rhoai/odh-pipeline-runtime-datascience-cpu-py312-rhel9`
- **Python packages**: `huggingface_hub>=0.20.0`
- **PVC**: Must be mounted by the pipeline (via `kubernetes.mount_pvc`) before this component runs

## Additional Resources

- **Documentation**: [https://github.com/kubeflow/pipelines-components](https://github.com/kubeflow/pipelines-components)
