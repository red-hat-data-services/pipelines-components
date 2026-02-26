# Open Shot Model Compressor ✨

> ⚠️ **Stability: alpha** — This asset is not yet stable and may change.

## Overview 🧾

Compress a causal language model using one-shot quantization.

Loads a Hugging Face causal LM and a calibration dataset, preprocesses and
tokenizes the data, then applies one-shot quantization via llmcompressor.
The compressed model and tokenizer are saved to the output artifact path.

## Inputs 📥

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_id` | `str` | `None` | Hugging Face model identifier (e.g. "meta-llama/Llama-3-8B"). |
| `quantization_scheme` | `str` | `None` | llmcompressor recipe(s) defining the compression strategy, e.g. W8A8, W4A16, NVFP4A16 etc. |
| `quantization_ignore_list` | `list[str]` | `None` | Layer names to exclude from quantization (e.g. ["lm_head"]). |
| `dataset_id` | `str` | `None` | Hugging Face dataset identifier used for calibration. |
| `dataset_split` | `str` | `None` | Dataset split to use (e.g. "train", "test"). |
| `output_model` | `dsl.Output[dsl.Artifact]` | `None` | Output artifact where the compressed model and tokenizer are saved. |
| `num_calibration_samples` | `int` | `512` | Number of dataset samples used for calibration (default: 512). |
| `max_sequence_length` | `int` | `2048` | Maximum token sequence length for truncation (default: 2048). |
| `seed` | `int` | `42` | Random seed for dataset shuffling (default: 42). |

## Metadata 🗂️

- **Name**: open_shot_model_compressor
- **Stability**: alpha
- **Dependencies**:
  - Kubeflow:
    - Name: Pipelines, Version: >=2.15.2
  - External Services:
    - Name: LLM Compressor, Version: 0.9.0.2
- **Tags**:
  - training
  - vllm
  - llm-compression
- **Last Verified**: 2026-02-26 04:00:42+00:00
- **Owners**:
  - Approvers: None
  - Reviewers: None

## Additional Resources 📚

- **Documentation**: [https://docs.vllm.ai/projects/llm-compressor](https://docs.vllm.ai/projects/llm-compressor)
