# Parse and Chunk Component

> **Stability: experimental** — This asset is not yet stable and may change.

KFP component that parses PDF documents using Docling and chunks them with HybridChunker via a distributed RayJob. Output chunks are written as JSONL files to an S3-compatible bucket (MinIO).

## How It Works

1. **Submit RayJob** -- Uses CodeFlare SDK (`ManagedClusterConfig` + `RayJob`) to create a Ray cluster with the specified number of workers. The entrypoint script is base64-encoded and passed as an environment variable to avoid needing ConfigMap/Secret permissions.
2. **Cluster Setup** -- Removes the Kueue queue-name label (if present) and unsuspends the RayJob so the cluster begins provisioning. Waits for all worker pods to reach `ready` state (up to 10 minutes).
3. **Distributed Processing** -- The Ray entrypoint script runs on the cluster:
   - Scans the PVC for PDF files at `{pvc_mount_path}/{input_path}`.
   - Creates a Ray Dataset from the file list and repartitions it across actors.
   - Each actor spawns a subprocess worker that initializes Docling's `DocumentConverter` and `HybridChunker`.
   - PDFs are converted to structured documents, then chunked into token-bounded text segments.
   - Each file's chunks are written as a JSONL file to S3 (`{s3_prefix}/{filename}.jsonl`).
4. **Monitoring** -- Progress is reported every 30 seconds with throughput stats and ETA.
5. **Completion** -- Polls the RayJob status until `SUCCEEDED` or `FAILED`.

### Actor Architecture

Each Ray actor (`DoclingChunkProcessor`) runs a dedicated subprocess to isolate Docling's heavy memory usage. If a file exceeds `timeout_seconds`, the subprocess is killed and restarted, allowing the actor to continue processing remaining files. This prevents a single bad PDF from blocking the entire pipeline.

### RayJob Management

All Kubernetes interactions (label removal, unsuspend, status polling) use the Kubernetes Python client (`CustomObjectsApi`), not CLI tools. This makes the component portable across environments without requiring `oc` or `kubectl`.

## Inputs

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pvc_name` | `str` | *required* | Name of the PVC containing input PDFs |
| `pvc_mount_path` | `str` | *required* | Mount path for the PVC inside pods |
| `input_path` | `str` | *required* | Relative path to PDF files under the PVC mount |
| `ray_image` | `str` | *required* | Container image with Ray + Docling pre-installed |
| `namespace` | `str` | *required* | OpenShift namespace for the RayJob |
| `s3_endpoint` | `str` | *required* | S3-compatible endpoint URL (e.g. MinIO) |
| `s3_bucket` | `str` | *required* | S3 bucket name for output chunks |
| `s3_prefix` | `str` | `"chunks"` | Key prefix for chunk files in S3 |
| `tokenizer` | `str` | `"sentence-transformers/all-MiniLM-L6-v2"` | Tokenizer for HybridChunker (typically the embedding model name) |
| `chunk_max_tokens` | `int` | `256` | Maximum tokens per chunk |
| `num_workers` | `int` | `2` | Number of Ray worker pods |
| `worker_cpus` | `int` | `8` | CPUs per Ray worker pod |
| `worker_memory_gb` | `int` | `16` | Memory (GB) per Ray worker pod |
| `head_cpus` | `int` | `2` | CPUs for the Ray head pod |
| `head_memory_gb` | `int` | `8` | Memory (GB) for the Ray head pod |
| `cpus_per_actor` | `int` | `4` | CPUs allocated per Docling processing actor |
| `min_actors` | `int` | `2` | Minimum actor pool size |
| `max_actors` | `int` | `4` | Maximum actor pool size |
| `batch_size` | `int` | `4` | Number of files per batch sent to each actor |
| `num_files` | `int` | `0` | Number of PDFs to process (0 = all) |
| `timeout_seconds` | `int` | `600` | Per-file processing timeout in seconds |
| `enable_profiling` | `bool` | `False` | Enable cProfile profiling output |
| `verbose` | `bool` | `True` | Enable verbose logging |

### Environment Variables (from Kubernetes Secret)

S3 credentials are injected from a Kubernetes Secret via the pipeline orchestrator (`kubernetes.use_secret_as_env`), not passed as direct parameters:

| Variable | Secret Key | Description |
|----------|------------|-------------|
| `S3_ACCESS_KEY` | `access_key` | S3/MinIO access key |
| `S3_SECRET_KEY` | `secret_key` | S3/MinIO secret key |

## Output

| Type | Description |
|------|-------------|
| `str` | S3 URI where JSONL chunk files were written (e.g. `s3://rag-chunks/chunks`) |

### JSONL Format

Each line in the output JSONL files contains:

```json
{"source_file": "document.pdf", "chunk_index": 0, "text": "The chunked text content..."}
{"source_file": "document.pdf", "chunk_index": 1, "text": "Second chunk text..."}
```

Files are written to `s3://{s3_bucket}/{s3_prefix}/{stem}.jsonl`.

## Metadata

- **Name**: parse_and_chunk
- **Stability**: experimental
- **Dependencies**:
  - Kubeflow:
    - Name: Pipelines, Version: >=2.15.2
  - External Services:
    - Name: KubeRay Operator, Version: >=1.4.0
- **Tags**: data_processing, pdf, docling, ray, chunking, rag
- **Last Verified**: 2026-04-28
- **Owners**:
  - Approvers: szaher

## Dependencies

- **Base image**: `registry.redhat.io/rhoai/odh-pipeline-runtime-datascience-cpu-py312-rhel9`
- **Python packages** (KFP pod): `codeflare-sdk==0.35.0`, `kubernetes>=28.1.0`
- **Ray runtime packages** (installed on the cluster): `opencv-python-headless`, `pypdfium2`, `boto3>=1.28.0`
- **Pre-installed in `ray_image`**: `docling`, `ray`, `pandas`
- **External services**: KubeRay Operator (>= 1.4.0), S3-compatible object store (MinIO)

## Additional Resources

- **Documentation**: [https://github.com/kubeflow/pipelines-components](https://github.com/kubeflow/pipelines-components)

## Usage

```python
from components.parse_and_chunk import parse_and_chunk

@dsl.pipeline(name="My Pipeline")
def my_pipeline():
    chunk_task = parse_and_chunk(
        pvc_name="data-pvc",
        pvc_mount_path="/mnt/data",
        input_path="input/pdfs",
        ray_image="quay.io/rhoai-szaher/docling-ray:latest",
        namespace="ray-docling",
        s3_endpoint="http://minio-service.default.svc.cluster.local:9000",
        s3_bucket="rag-chunks",
    )
    # Inject S3 credentials from a Kubernetes Secret
    kubernetes.use_secret_as_env(
        chunk_task, secret_name="minio-secret",
        secret_key_to_env={"access_key": "S3_ACCESS_KEY", "secret_key": "S3_SECRET_KEY"},
    )
```
