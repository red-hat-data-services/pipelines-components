<!-- markdownlint-disable MD013 -->
# Test Data Loader 📊

> ⚠️ **Stability: alpha** — This asset is not yet stable and may change.

## Overview 🧾

Reads test data from JSON files using a DataLoader component.

The Test Data Loader component loads test data from JSON files for AutoRAG evaluation. It reads
test data from various sources including S3 (via RHOAI Connections API) or local filesystem and
returns the data as a pandas DataFrame. This test data is used for evaluating RAG configurations
during the optimization process.

The component supports JSON format only and is typically the first component executed in the
AutoRAG pipeline, as the test data is needed for document sampling in subsequent steps.

## Inputs 📥

| Parameter               | Type                       | Default   | Description                                                            |
|-------------------------|----------------------------|-----------|------------------------------------------------------------------------|
| `test_data`             | `dsl.Output[dsl.Artifact]` | `None`    | Output artifact containing the loaded test data as a pandas DataFrame. |
| `test_data_bucket_name` | `str`                      | Mandatory | S3 bucket that contains the test data file.                            |
| `test_data_path`        | `str`                      | Mandatory | S3 object key to the JSON test data file.                              |

### Test Data Reference Structure

To access the test data stored in an S3-compatible storage, the component requires the following environment variables to be available at runtime:

| Environment variable name | Description                                         |
|--------------------------|-----------------------------------------------------|
| `AWS_ACCESS_KEY_ID`      | access key used to authenticate with the S3 service |
| `AWS_SECRET_ACCESS_KEY`  | secret key used to authenticate with the S3 service |
| `AWS_S3_ENDPOINT`        | endpoint URL of the S3 instance                     |
| `AWS_REGION`             | region in which the S3 instance is deployed         |

## Outputs 📤

| Output      | Type           | Description                                          |
|-------------|----------------|------------------------------------------------------|
| `test_data` | `dsl.Artifact` | The loaded test data as a pandas DataFrame artifact. |

## Usage Examples 💡

### Basic Usage

```python
from kfp import dsl
from kfp_components.components.data_processing.autorag.test_data_loader import test_data_loader

@dsl.pipeline(name="test-data-loading-pipeline")
def my_pipeline():
    """Example pipeline demonstrating test data loading."""
    load_task = test_data_loader(
        test_data_bucket_name="s3-test-data-bucket",
        test_data_path="s3-test-data-path"
    )
    return load_task
```

### Local Filesystem

```python
@dsl.pipeline(name="local-test-data-loading-pipeline")
def my_pipeline():
    """Example pipeline loading from local filesystem."""
    load_task = test_data_loader(
        test_data_bucket_name="s3-test-data-bucket",
        test_data_path="s3-test-data-path"
    )
    return load_task
```

## Test Data Format 📋

The component expects test data in JSON format. The JSON file should contain test questions and
expected answers for RAG evaluation.

## Notes 📝

- **JSON Format Only**: Only JSON format is supported for test data files
- **DataFrame Output**: Returns data as a pandas DataFrame for easy processing
- **Early Execution**: Typically executed first in the pipeline as test data is needed for document
  sampling
- **Connection Management**: Uses RHOAI Connections API for secure access to S3 and other cloud
  storage systems

## Metadata 🗂️

- **Name**: test_data_loader
- **Stability**: alpha
- **Dependencies**:
  - Kubeflow:
    - Name: Pipelines, Version: >=2.15.2
  - External Services:
    - Name: pandas, Version: >=2.0.0
- **Tags**:
  - data-processing
  - autorag
  - test-data
- **Last Verified**: 2026-01-23 00:00:00+00:00

## Additional Resources 📚

- **AutoRAG Documentation**: See AutoRAG pipeline documentation for comprehensive information
- **Issue Tracker**: [GitHub Issues](https://github.com/kubeflow/pipelines-components/issues)
