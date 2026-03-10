<!-- markdownlint-disable MD013 -->
# Text Extraction 📝

> ⚠️ **Stability: alpha** — This asset is not yet stable and may change.

## Overview 🧾

Reads the sampled-documents descriptor (YAML) produced by the documents_sampling, fetches those
documents from S3, and extracts text using the docling library.

The Text Extraction component takes the **sampled_documents_descriptor** artifact (a YAML manifest
from documents_sampling), downloads the listed documents from S3, and extracts text from them. It
supports PDF, DOCX, PPTX, Markdown, HTML, and plain text. The extracted text is written as
markdown for downstream steps (chunking, embedding). This component requires S3 credentials at
runtime to fetch the documents described in the descriptor.

## Inputs 📥

| Parameter                      | Type                       | Default  | Description                                                          |
|--------------------------------|----------------------------|----------|----------------------------------------------------------------------|
| `sampled_documents_descriptor` | `dsl.Input[dsl.Artifact]`   | Mandatory| Input artifact containing `sampled_documents_descriptor.yaml` from documents_sampling. |
| `extracted_text`               | `dsl.Output[dsl.Artifact]` | `None`   | Output artifact where extracted text (markdown) is written.          |

### S3 credentials (for fetching documents)

The component downloads documents from S3 using the bucket and keys in the descriptor. Set these environment variables (e.g. via a Kubernetes secret) when running the component:

| Environment variable name | Description                                         |
|---------------------------|-----------------------------------------------------|
| `AWS_ACCESS_KEY_ID`       | Access key for the S3 service.                      |
| `AWS_SECRET_ACCESS_KEY`   | Secret key for the S3 service.                      |
| `AWS_S3_ENDPOINT`         | Endpoint URL of the S3 instance.                    |
| `AWS_REGION`              | Region of the S3 instance.                         |

## Outputs 📤

| Output           | Type           | Description                                                           |
|------------------|----------------|-----------------------------------------------------------------------|
| `extracted_text` | `dsl.Artifact` | Extracted text as markdown files (one per document), for chunking and embedding. |

## Usage Examples 💡

### Basic Usage

```python
from kfp import dsl
from kfp_components.components.data_processing.autorag.text_extraction import text_extraction

@dsl.pipeline(name="text-extraction-pipeline")
def my_pipeline(sampled_documents_descriptor):
    """Example pipeline demonstrating text extraction."""
    extract_task = text_extraction(
        sampled_documents_descriptor=sampled_documents_descriptor
    )
    return extract_task
```

### In AutoRAG Pipeline Context

```python
@dsl.pipeline(name="autorag-pipeline")
def autorag_pipeline(documents_sampling_task):
    """Example AutoRAG pipeline with text extraction."""
    # documents_sampling_task.outputs["sampled_documents"] contains the descriptor YAML
    extract_task = text_extraction(
        sampled_documents_descriptor=documents_sampling_task.outputs["sampled_documents"]
    )
    # Attach S3 secret to extract_task so it can download documents
    # ...
    return extract_task
```

## Supported Document Formats 📋

The component supports extraction from the following document formats:

- **PDF** (`.pdf`) - Portable Document Format
- **DOCX** (`.docx`) - Microsoft Word documents
- **PPTX** (`.pptx`) - Microsoft PowerPoint presentations
- **Markdown** (`.md`) - Markdown files
- **HTML** (`.html`) - HTML documents
- **Plain text** (`.txt`) - Text files

## Docling Library 🔧

The component uses the `docling` library for text extraction. Docling provides:

- High-quality text extraction from various document formats
- Preservation of document structure and formatting where applicable
- Support for complex document layouts
- Robust handling of different document types

## Notes 📝

- **Descriptor input**: Expects the artifact produced by documents_sampling (directory containing `sampled_documents_descriptor.yaml`).
- **S3 fetch**: Downloads documents from S3 using bucket/keys in the descriptor; the pipeline must provide S3 credentials to this component.
- **Format support**: Handles PDF, DOCX, PPTX, MD, HTML, and TXT.
- **Output**: One markdown file per document, suitable for chunking and embedding.

## Metadata 🗂️

- **Name**: text_extraction
- **Stability**: alpha
- **Dependencies**:
  - Kubeflow:
    - Name: Pipelines, Version: >=2.15.2
  - External Services:
    - Name: docling, Version: >=1.0.0
- **Tags**:
  - data-processing
  - autorag
  - text-extraction
- **Last Verified**: 2026-01-23 00:00:00+00:00

## Additional Resources 📚

- **AutoRAG Documentation**: See AutoRAG pipeline documentation for comprehensive information
- **Issue Tracker**: [GitHub Issues](https://github.com/kubeflow/pipelines-components/issues)
