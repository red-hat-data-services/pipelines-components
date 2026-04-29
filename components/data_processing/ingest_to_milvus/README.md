# Ingest to Milvus Component

> **Stability: experimental** — This asset is not yet stable and may change.

KFP component that reads chunked text from S3, generates vector embeddings, and inserts them into a Milvus collection. Supports two embedding modes: local sentence-transformers model or a deployed embedding service endpoint.

## How It Works

1. **Read Chunks from S3** -- Uses boto3 to paginate through all `.jsonl` files under `{s3_bucket}/{s3_prefix}/`. Each line is parsed as a JSON object with `source_file`, `chunk_index`, and `text` fields. Raises `FileNotFoundError` if no chunks are found.
2. **Create Milvus Collection** -- Drops any existing collection with the same name, then creates a new one with fields for id, source_file, chunk_index, text, and embedding. Creates an IVF_FLAT index with COSINE metric (`nlist=128`).
3. **Generate Embeddings** -- Two modes based on the `embedding_endpoint` parameter:
   - **Local mode** (empty string): Loads the sentence-transformers model in-process and encodes texts with normalized embeddings.
   - **Service mode** (URL provided): Sends batched POST requests to `{embedding_endpoint}/v1/embeddings` using the OpenAI-compatible API. Responses are sorted by index to maintain ordering.
4. **Batch Insert** -- Inserts vectors into Milvus in configurable batch sizes (`milvus_batch_size`). Progress is logged every 10 batches.
5. **Finalize** -- Loads the collection for searching and prints collection stats.

## Inputs

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `s3_endpoint` | `str` | *required* | S3-compatible endpoint URL (e.g. MinIO) |
| `s3_bucket` | `str` | *required* | S3 bucket containing chunk files |
| `milvus_host` | `str` | *required* | Milvus service hostname |
| `s3_prefix` | `str` | `"chunks"` | Key prefix for chunk files in S3 |
| `embedding_endpoint` | `str` | `""` | Embedding service URL. Empty = use local model |
| `embedding_model` | `str` | `"sentence-transformers/all-MiniLM-L6-v2"` | Model name (for local or API mode) |
| `embedding_dim` | `int` | `384` | Dimension of the embedding vectors |
| `milvus_port` | `int` | `19530` | Milvus gRPC port |
| `milvus_db` | `str` | `"default"` | Milvus database name |
| `collection_name` | `str` | `"rag_documents"` | Milvus collection name to create |
| `embed_batch_size` | `int` | `64` | Batch size for embedding generation |
| `milvus_batch_size` | `int` | `256` | Batch size for Milvus inserts |

### Environment Variables (from Kubernetes Secret)

S3 credentials are injected from a Kubernetes Secret via the pipeline orchestrator (`kubernetes.use_secret_as_env`):

| Variable | Secret Key | Description |
|----------|------------|-------------|
| `S3_ACCESS_KEY` | `access_key` | S3/MinIO access key |
| `S3_SECRET_KEY` | `secret_key` | S3/MinIO secret key |

## Output

| Type | Description |
|------|-------------|
| `str` | Collection name and total vectors inserted (e.g. `rag_documents:15234`) |

### Milvus Collection Schema

| Field | Type | Description |
|-------|------|-------------|
| `id` | INT64 | Auto-generated primary key |
| `source_file` | VARCHAR(512) | Original PDF filename |
| `chunk_index` | INT64 | Chunk position within the source file |
| `text` | VARCHAR(32768) | Chunk text content |
| `embedding` | FLOAT_VECTOR | Vector embedding (dimension = `embedding_dim`) |

**Index**: IVF_FLAT with COSINE metric, nlist=128.

## Embedding Modes

**Local mode** (`embedding_endpoint = ""`):
- Loads `sentence-transformers` model directly in the component pod
- Simpler setup, no external service dependency
- Requires more memory in the pipeline pod
- Good for testing or small datasets

**Service mode** (`embedding_endpoint = "http://..."`) :
- Calls an OpenAI-compatible `/v1/embeddings` API
- Works with TEI (Text Embeddings Inference), or any compatible embedding service
- Better resource utilization, can leverage GPU on the embedding service
- Ideal for production workloads

## Metadata

- **Name**: ingest_to_milvus
- **Stability**: experimental
- **Dependencies**:
  - Kubeflow:
    - Name: Pipelines, Version: >=2.15.2
  - External Services:
    - Name: Milvus, Version: >=2.4.0
- **Tags**: milvus, embeddings, vector_db, ingestion, rag
- **Last Verified**: 2026-04-28
- **Owners**:
  - Approvers: szaher

## Dependencies

- **Base image**: `registry.redhat.io/rhoai/odh-pipeline-runtime-datascience-cpu-py312-rhel9`
- **Python packages**: `pymilvus>=2.4.0`, `sentence-transformers>=2.2.0`, `requests>=2.28.0`, `boto3>=1.28.0`
- **External services**: Milvus (>= 2.4.0), S3-compatible object store

## Additional Resources

- **Documentation**: [https://github.com/kubeflow/pipelines-components](https://github.com/kubeflow/pipelines-components)

## Usage

```python
from components.ingest_to_milvus import ingest_to_milvus

@dsl.pipeline(name="My Pipeline")
def my_pipeline():
    ingest_task = ingest_to_milvus(
        s3_endpoint="http://minio-service.default.svc.cluster.local:9000",
        s3_bucket="rag-chunks",
        milvus_host="milvus-milvus.milvus.svc.cluster.local",
        embedding_endpoint="",  # empty = local, URL = service mode
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    )
    # Inject S3 credentials from a Kubernetes Secret
    kubernetes.use_secret_as_env(
        ingest_task, secret_name="minio-secret",
        secret_key_to_env={"access_key": "S3_ACCESS_KEY", "secret_key": "S3_SECRET_KEY"},
    )
```
