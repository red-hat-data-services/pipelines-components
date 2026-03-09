# Documents Indexing 📇

> ⚠️ **Stability: alpha** — This asset is not yet stable and may change.

## Overview

Reads markdown files from the **extracted_text** artifact (produced by the [Text Extraction](../text_extraction/README.md) component),
chunks them, embeds them via Llama Stack, and indexes them into a vector store. Supports **batch processing**: when `batch_size` > 0, documents are read, chunked, embedded, and indexed in batches to limit memory use and allow progress on large inputs.

## Inputs

| Parameter                        | Type                      | Default       | Description                                                                      |
|----------------------------------|---------------------------|---------------|----------------------------------------------------------------------------------|
| `embedding_params`               | `dict`                    | —             | Embedding parameters.                                                            |
| `embedding_model_id`             | `str`                     | —             | Embedding model ID.                                                              |
| `extracted_text`                 | `dsl.Input[dsl.Artifact]` | —             | Input artifact (folder) containing `.md` files from text extraction.             |
| `llama_stack_vector_database_id` | `str`                     | —             | Vector store provider ID.                                                        |
| `distance_metric`                | `str`                     | `"cosine"`    | Vector distance metric.                                                          |
| `chunking_method`                | `str`                     | `"recursive"` | Chunking method (e.g. LangChain recursive splitter).                             |
| `chunk_size`                     | `int`                     | `1024`        | Chunk size in characters.                                                        |
| `chunk_overlap`                  | `int`                     | `0`           | Chunk overlap in characters.                                                     |
| `batch_size`                     | `int`                     | `50`          | Number of documents per batch. Set to `0` to process all documents in one batch. |

### Llama Stack credentials

The component uses Llama Stack for embedding and vector store. Set these environment variables (e.g. via a Kubernetes secret) when running the component:

| Environment variable             | Description                      |
|----------------------------------|----------------------------------|
| `LLAMA_STACK_CLIENT_BASE_URL`    | Base URL of the Llama Stack API. |
| `LLAMA_STACK_CLIENT_API_KEY`     | API key for the Llama Stack API. |

## Outputs

This component does not produce pipeline artifacts. It writes embedded chunks directly to the vector store configured via `llama_stack_vector_database_id` and the Llama Stack client.

## Batch processing

- **`batch_size > 0`** (default `20`): Documents are processed in batches. For each batch the component reads files, chunks them, embeds, and calls the vector store. This reduces peak memory and surfaces progress in logs.
- **`batch_size <= 0`**: All documents are processed in a single batch (legacy behavior).

## Dependencies

- **ai4rag** (embedding, chunking, vector store)
- **langchain-text-splitters** (chunking)
- **llama_stack_client** (Llama Stack API)
