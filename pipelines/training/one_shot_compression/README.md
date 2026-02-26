# One Shot Compression ✨

> ⚠️ **Stability: alpha** — This asset is not yet stable and may change.

## Overview 🧾

Compress a model with LLMCompressor and register it in the model registry.

Runs one-shot quantization on a Hugging Face causal LM using the specified
calibration dataset and quantization scheme, then registers the compressed
model artifact in a Kubeflow Model Registry instance.

## Inputs 📥

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_id` | `str` | `ibm-granite/granite-3.3-8b-instruct` | Hugging Face model identifier to compress. |
| `dataset_id` | `str` | `HuggingFaceH4/ultrachat_200k` | Hugging Face dataset identifier for calibration. |
| `dataset_split` | `str` | `train_sft` | Dataset split to use for calibration. |
| `quantization_scheme` | `str` | `W4A16` | Quantization scheme to apply (e.g. "W4A16"). |
| `quantization_ignore_list` | `list[str]` | `['lm_head']` | Layer names to exclude from quantization. |
| `cpu_requests` | `str` | `3000m` | CPU resource request for the compression task. |
| `memory_request` | `str` | `75G` | Memory resource request for the compression task. |
| `accelerator_type` | `str` | `nvidia.com/gpu` | Accelerator resource type (e.g. "nvidia.com/gpu"). |
| `accelerator_limit` | `str` | `1` | Number of accelerator devices to request. |
| `model_registry_address` | `str` | `` | Address of the Model Registry server. |
| `model_registry_author` | `str` | `pipeline` | Author name for the registered model. |
| `model_registry_model_name` | `str` | `compressed-model` | Name to register the compressed model under. |
| `model_registry_model_version` | `str` | `1.0.0` | Version string for the registered model. |
| `model_registry_opt_description` | `str` | `Compressed Model via one shot compression` | Description for the registered model. |
| `model_registry_opt_format_name` | `str` | `pytorch` | Serialization format of the model. |
| `model_registry_opt_port` | `int` | `8080` | Port of the Model Registry server. |

## Metadata 🗂️

- **Name**: one_shot_compression
- **Stability**: alpha
- **Dependencies**:
  - Kubeflow:
    - Name: Pipelines, Version: >=2.15.2
- **Tags**:
  - training
  - pipeline
- **Last Verified**: 2026-02-26 20:14:19+00:00
- **Owners**:
  - Approvers:
    - nsingla
  - Reviewers: None
