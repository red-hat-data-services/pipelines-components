# PDF Documents Processing RAG Pipeline

> **Stability: experimental** — This asset is not yet stable and may change.

## Overview

Multi-step RAG pipeline that parses and chunks PDF documents using Docling via
RayJob, ingests embeddings into Milvus, and deploys an LLM for inference using
vLLM on OpenShift AI.

The pipeline orchestrates five reusable components in two parallel chains:

- **Data chain**: parse_and_chunk -> ingest_to_milvus
- **Model chain**: download_model -> model_deployment
- **Optional**: deploy_embedding_model (when deploy_embedding=True)

## Inputs

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pvc_name` | `str` | `"data-pvc"` | PVC containing input PDFs |
| `pvc_mount_path` | `str` | `"/mnt/data"` | Mount path for the data PVC |
| `namespace` | `str` | `"ray-docling"` | OpenShift namespace |
| `s3_endpoint` | `str` | `"http://minio-service.default.svc.cluster.local:9000"` | S3/MinIO endpoint |
| `s3_bucket` | `str` | `"rag-chunks"` | S3 bucket for intermediate chunks |
| `s3_prefix` | `str` | `"chunks"` | S3 key prefix for chunk files |
| `s3_secret_name` | `str` | `"minio-secret"` | Kubernetes Secret with S3 credentials |
| `input_path` | `str` | `"input/pdfs"` | Path to PDFs on the PVC |
| `ray_image` | `str` | `"quay.io/rhoai-szaher/docling-ray:latest"` | Ray + Docling container image |
| `num_workers` | `int` | `2` | Ray worker pods |
| `deploy_embedding` | `bool` | `False` | Deploy embedding model as InferenceService |
| `embedding_endpoint` | `str` | `""` | Embedding service URL (empty = local model) |
| `embedding_model` | `str` | `"ibm-granite/granite-embedding-125m-english"` | Embedding model name |
| `embedding_dim` | `int` | `768` | Embedding vector dimension |
| `milvus_host` | `str` | `"milvus-milvus.milvus.svc.cluster.local"` | Milvus hostname |
| `collection_name` | `str` | `"rag_documents"` | Milvus collection name |
| `llm_model_name` | `str` | `"mistralai/Mistral-7B-Instruct-v0.3"` | LLM model for inference |
| `model_cache_pvc` | `str` | `"model-cache-pvc"` | PVC for cached model weights |
| `gpu_count` | `int` | `1` | GPUs for LLM serving |

## Components Used

| Component | Category | Description |
|-----------|----------|-------------|
| [parse_and_chunk](../../../../components/data_processing/parse_and_chunk/README.md) | data_processing | Parse PDFs with Docling via RayJob, chunk with HybridChunker, write JSONL to S3 |
| [ingest_to_milvus](../../../../components/data_processing/ingest_to_milvus/README.md) | data_processing | Read chunks from S3, generate embeddings, insert into Milvus |
| [download_model](../../../../components/data_processing/download_model/README.md) | data_processing | Download HuggingFace model to PVC with caching |
| [deploy_embedding_model](../../../../components/deployment/deploy_embedding_model/README.md) | deployment | Deploy embedding model as KServe InferenceService (optional) |
| [model_deployment](../../../../components/deployment/model_deployment/README.md) | deployment | Deploy LLM with vLLM on KServe from PVC cache |

## Metadata

- **Name**: pdf_documents_processing_rag
- **Stability**: experimental
- **Dependencies**:
  - Kubeflow:
    - Name: Pipelines, Version: >=2.15.2
  - External Services:
    - Name: KubeRay Operator, Version: >=1.4.0
    - Name: Milvus, Version: >=2.4.0
    - Name: KServe, Version: >=0.11.0
- **Tags**: data_processing, pdf, docling, ray, chunking, rag, milvus
- **Last Verified**: 2026-04-28
- **Owners**:
  - Approvers: szaher

## Additional Resources

- **Documentation**: [https://github.com/kubeflow/pipelines-components](https://github.com/kubeflow/pipelines-components)
