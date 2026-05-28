# Pipeline Manifest Publisher ✨

> ⚠️ **Stability: alpha** — This asset is not yet stable and may change.

## Overview 🧾

Publish the pipeline structure manifest for dashboard consumption.

Reads the static JSON manifest from the package (run_status_templates/pipelines/) and publishes it as a KFP artifact. This enables dashboards to show expected components, stages, and steps before pipeline execution begins.

The manifest defines the complete pipeline structure: - Component list and execution order - Stages within each component - Steps within each stage (optional) - Descriptions for UI display

## Inputs 📥

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `pipeline_id` | `str` | `None` | Pipeline identifier matching manifest filename (e.g., "autogluon-tabular-training-pipeline"). |
| `run_id` | `str` | `None` | KFP run ID for tracking (from dsl.PIPELINE_JOB_ID_PLACEHOLDER). |
| `pipeline_manifest` | `dsl.Output[dsl.Artifact]` | `None` | Output artifact containing the full pipeline structure. |

## Outputs 📤

| Name | Type | Description |
| ---- | ---- | ----------- |
| Output | `None` |  |

## Metadata 🗂️

- **Name**: pipeline_manifest_publisher
- **Stability**: alpha
- **Dependencies**:
  - Kubeflow:
    - Name: Pipelines, Version: >=2.15.2
- **Tags**:
  - automl
  - run-status
- **Last Verified**: 2026-05-28 00:00:00+00:00
- **Owners**:
  - Approvers:
    - LukaszCmielowski
    - DorotaDR
  - Reviewers:
    - Mateusz-Switala
    - DorotaDR
