# AutoRAG Documents Indexing Pipeline

> ⚠️ **Stability: alpha** — This asset is not yet stable and may change.

## Overview

Extends the [AutoRAG Data Processing Pipeline](../documents_processing_pipeline/README.md) with a **documents indexing** step.
Loads test data without sampling, extracts text, then chunks and indexes the extracted text into a vector store (Llama Stack).

## Pipeline workflow

1**Documents discovery** — Lists documents in the input S3 bucket/prefix without sampling, writes a descriptor json.
2**Text extraction** — Fetches the listed documents from S3 and extracts text using docling; outputs markdown files.
3**Documents indexing** — Chunks the extracted text, embeds via Llama Stack, and writes to a vector store collection.

## Inputs

| Parameter                        | Type              | Default       | Description                                              |
|----------------------------------|-------------------|---------------|----------------------------------------------------------|
| `input_data_secret_name`         | `str`             | —             | Secret with S3 credentials for input data.               |
| `input_data_bucket_name`         | `str`             | —             | S3 bucket containing input documents.                    |
| `input_data_key`                 | `str`             | —             | Path to folder with input documents in the bucket.       |
| `llama_stack_secret_name`        | `str`             | —             | Secret with Llama stack credentials                      |
| `embedding_model_id`             | `str`             | —             | Embedding model ID for the vector store.                 |
| `llama_stack_vector_database_id` | `Optional[str]`   | `None`        | Vector store provider ID.                                |
| `collection_name`                | `str`             | —             | Name of the vector store collection.                     |
| `embedding_params`               | `dict`            | `{}`          | Optional embedding model parameters                      |
| `distance_metric`                | `str`             | `"cosine"`    | Vector distance metric.                                  |
| `chunking_method`                | `str`             | `"recursive"` | Chunking method.                                         |
| `chunk_size`                     | `int`             | `1024`        | Chunk size in characters.                                |
| `chunk_overlap`                  | `int`             | `0`           | Chunk overlap in characters.                             |
| `batch_size`                     | `int`             | `20`          | Number of documents per batch (0 = process all at once). |

### S3 credentials (for fetching documents)

The component downloads documents from S3, set these environment variables (e.g. via a Kubernetes secret) when running the component:

| Environment variable name | Description                          |
|---------------------------|--------------------------------------|
| `AWS_ACCESS_KEY_ID`       | Access key for the S3 service.       |
| `AWS_SECRET_ACCESS_KEY`   | Secret key for the S3 service.       |
| `AWS_S3_ENDPOINT`         | Endpoint URL of the S3 instance.     |
| `AWS_DEFAULT_REGION`      | Region of the S3 instance.           |

### Llama Stack credentials

The component uses Llama Stack for embedding and vector store. Set these environment variables (e.g. via a Kubernetes secret) when running the component:

| Environment variable             | Description                      |
|----------------------------------|----------------------------------|
| `LLAMA_STACK_CLIENT_BASE_URL`    | Base URL of the Llama Stack API. |
| `LLAMA_STACK_CLIENT_API_KEY`     | API key for the Llama Stack API. |

## Components used

1. [Test data loader](../../../components/data_processing/autorag/test_data_loader/README.md)
2. [Documents discovery](../../../components/data_processing/autorag/documents_discovery/README.md)
3. [Text extraction](../../../components/data_processing/autorag/text_extraction/README.md)
4. [Documents indexing](../../../components/data_processing/autorag/documents_indexing/README.md)

## Compiling the pipeline

From the repo root:

```bash
python pipelines/data_processing/autorag/documents_indexing_pipeline/pipeline.py
```

This produces `documents_indexing_pipeline.yaml` in the same directory.
