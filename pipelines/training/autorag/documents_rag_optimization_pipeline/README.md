# Documents Rag Optimization Pipeline ✨

> ⚠️ **Stability: alpha** — This asset is not yet stable and may change.

## Overview 🧾

A simple test pipeline.

This pipeline demonstrates basic pipeline structure for testing the README generator.

## Inputs 📥

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_text` | `str` | `None` | The input text to process. |
| `iterations` | `int` | `3` | Number of iterations to run. |

## Metadata 🗂️

- **Name**: documents_rag_optimization_pipeline
- **Stability**: alpha
- **Dependencies**:
  - Kubeflow:
    - Name: Pipelines, Version: >=2.15.2
  - External Services:
    - Name: ai4rag, Version: >=1.0.0
    - Name: llama-stack API, Version: >=1.0.0
    - Name: RHOAI Connections API, Version: >=1.0.0
    - Name: Milvus, Version: >=2.0.0
    - Name: Milvus Lite, Version: >=2.0.0
    - Name: MLFlow, Version: >=2.0.0
    - Name: docling, Version: >=1.0.0
- **Tags**:
  - training
  - pipeline
  - autorag
  - rag-optimization
- **Last Verified**: 2026-03-09 13:14:31+00:00
- **Owners**:
  - Approvers:
    - HumairAK
    - mprahl
