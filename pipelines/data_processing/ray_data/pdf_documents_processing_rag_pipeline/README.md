# Pdf Documents Processing Rag ✨

> ⚠️ **Stability: experimental** — This asset is not yet stable and may change.

## Overview 🧾

Multi-step RAG pipeline: parse PDFs, ingest into Milvus, deploy LLM.

## Inputs 📥

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `pvc_name` | `str` | `data-pvc` | PVC containing input PDF documents. |
| `pvc_mount_path` | `str` | `/mnt/data` | Mount path for the data PVC inside pods. |
| `namespace` | `str` | `ray-docling` | OpenShift namespace for all resources. |
| `s3_endpoint` | `str` | `http://minio-service.default.svc.cluster.local:9000` | S3-compatible endpoint URL (e.g. MinIO). |
| `s3_bucket` | `str` | `rag-chunks` | S3 bucket for intermediate chunk storage. |
| `s3_prefix` | `str` | `chunks` | Key prefix for chunk files in S3. |
| `s3_secret_name` | `str` | `minio-secret` | Kubernetes Secret with S3 credentials. |
| `input_path` | `str` | `input/pdfs` | Path to PDF files on the PVC. |
| `ray_image` | `str` | `quay.io/rhoai-szaher/docling-ray:latest` | Container image with Ray and Docling pre-installed. |
| `num_workers` | `int` | `2` | Number of Ray worker pods. |
| `worker_cpus` | `int` | `8` | CPUs per Ray worker pod. |
| `worker_memory_gb` | `int` | `16` | Memory (GB) per Ray worker pod. |
| `head_cpus` | `int` | `2` | CPUs for the Ray head pod. |
| `head_memory_gb` | `int` | `8` | Memory (GB) for the Ray head pod. |
| `cpus_per_actor` | `int` | `4` | CPUs per Docling processing actor. |
| `min_actors` | `int` | `2` | Minimum actor pool size. |
| `max_actors` | `int` | `4` | Maximum actor pool size. |
| `batch_size` | `int` | `4` | Files per batch sent to each actor. |
| `chunk_max_tokens` | `int` | `256` | Maximum tokens per chunk. |
| `num_files` | `int` | `1000` | Number of PDFs to process (0 = all). |
| `timeout_seconds` | `int` | `600` | Per-file processing timeout in seconds. |
| `enable_profiling` | `bool` | `False` | Enable cProfile profiling output. |
| `verbose` | `bool` | `True` | Enable verbose logging. |
| `bypass_kueue` | `bool` | `False` | If True, bypass Kueue quota management for the RayJob. |
| `deploy_embedding` | `bool` | `False` | If True, deploy embedding model as InferenceService. |
| `embedding_endpoint` | `str` | `""` | Embedding service URL (empty = local model). |
| `embedding_model` | `str` | `ibm-granite/granite-embedding-125m-english` | Embedding model name. |
| `embedding_dim` | `int` | `768` | Embedding vector dimension. |
| `embedding_runtime_image` | `str` | `_DEFAULT_EMBEDDING_RUNTIME_IMAGE` | Container image for the embedding server. |
| `embedding_gpu_count` | `int` | `1` | GPUs for the embedding service. |
| `milvus_host` | `str` | `milvus-milvus.milvus.svc.cluster.local` | Milvus service hostname. |
| `milvus_port` | `int` | `19530` | Milvus gRPC port. |
| `milvus_db` | `str` | `default` | Milvus database name. |
| `collection_name` | `str` | `rag_documents` | Milvus collection name. |
| `drop_existing` | `bool` | `True` | If True, drop and recreate the Milvus collection. If False, append. |
| `embed_batch_size` | `int` | `64` | Batch size for embedding requests. |
| `milvus_batch_size` | `int` | `256` | Batch size for Milvus inserts. |
| `hf_secret_name` | `str` | `hf-token-secret` | Kubernetes Secret with HuggingFace token (key: `token`). Required for gated models. |
| `llm_model_name` | `str` | `mistralai/Mistral-7B-Instruct-v0.3` | HuggingFace LLM model ID for inference. |
| `model_cache_pvc` | `str` | `model-cache-pvc` | PVC for cached model weights. |
| `max_model_len` | `int` | `4096` | Maximum context length for the LLM. |
| `gpu_count` | `int` | `1` | GPUs for LLM serving. |

## Prerequisites

The default LLM (`mistralai/Mistral-7B-Instruct-v0.3`) is a gated model on HuggingFace. Before running the pipeline, create a Kubernetes Secret with your HuggingFace token:

```bash
oc create secret generic hf-token-secret --from-literal=token=hf_xxxxx
```

The secret name must match the `hf_secret_name` pipeline parameter (default: `hf-token-secret`), and the key must be `token`.

## Metadata 🗂️

- **Name**: pdf_documents_processing_rag
- **Description**: Multi-step RAG pipeline that parses and chunks PDF documents using Docling via RayJob, ingests embeddings into Milvus, and deploys an LLM for inference using vLLM on OpenShift AI.

- **Stability**: experimental
- **Dependencies**:
  - Kubeflow:
    - Name: Pipelines, Version: >=2.15.2
  - External Services:
    - Name: KubeRay Operator, Version: >=1.4.0
    - Name: Milvus, Version: >=2.4.0
    - Name: KServe, Version: >=0.11.0
- **Tags**:
  - data_processing
  - pdf
  - docling
  - ray
  - chunking
  - rag
  - milvus
- **Last Verified**: 2026-04-28 00:00:00+00:00
- **Owners**:
  - Approvers:
    - szaher
  - Reviewers:
    - szaher

## Additional Resources 📚

- **Documentation**: [https://github.com/kubeflow/pipelines-components](https://github.com/kubeflow/pipelines-components)
