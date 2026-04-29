# Deploy Embedding Model Component

> **Stability: experimental** — This asset is not yet stable and may change.

KFP component that deploys a text embedding model as a KServe InferenceService on OpenShift AI. Uses HuggingFace Text Embeddings Inference (TEI) as the serving runtime, exposing an OpenAI-compatible `/v1/embeddings` API.

## How It Works

1. **Derive Names** -- Converts the HuggingFace model ID into a Kubernetes-safe InferenceService name (e.g. `sentence-transformers/all-MiniLM-L6-v2` becomes `all-minilm-l6-v2`).
2. **Create ServingRuntime** -- Creates or updates a `ServingRuntime` CR that defines the TEI container image, model ID, port, and resource limits. The TEI container downloads the model automatically at startup via the `--model-id` argument.
3. **Create InferenceService** -- Creates or updates an `InferenceService` CR that references the ServingRuntime. Uses `RawDeployment` mode (no Knative dependency). The model format is `huggingface` to match the ServingRuntime's `supportedModelFormats`.
4. **Wait for Ready** -- Polls the InferenceService status every 15 seconds (up to 15 minutes) until the `Ready` condition is `True`.
5. **Return Endpoint** -- Returns the service URL from the InferenceService status.

### Create-or-Update Pattern

Both the ServingRuntime and InferenceService use a GET-then-PATCH/CREATE pattern: if the resource already exists it is patched in place, otherwise it is created. This makes the component idempotent across pipeline reruns.

## Inputs

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name` | `str` | *required* | HuggingFace model ID (e.g. `sentence-transformers/all-MiniLM-L6-v2`) |
| `namespace` | `str` | *required* | Namespace to deploy the InferenceService |
| `serving_runtime_name` | `str` | `"embedding-runtime"` | Name of the ServingRuntime CR |
| `min_replicas` | `int` | `1` | Minimum number of replicas |
| `max_replicas` | `int` | `1` | Maximum number of replicas |
| `cpu_requests` | `str` | `"2"` | CPU requests per replica |
| `cpu_limits` | `str` | `"4"` | CPU limits per replica |
| `memory_requests` | `str` | `"4Gi"` | Memory requests per replica |
| `memory_limits` | `str` | `"8Gi"` | Memory limits per replica |

## Output

| Type | Description |
|------|-------------|
| `str` | The embedding service endpoint URL (e.g. `http://all-minilm-l6-v2.ray-docling.svc.cluster.local`) |

### API Usage

Once deployed, the service exposes an OpenAI-compatible embeddings API:

```bash
curl -X POST http://<endpoint>/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model": "sentence-transformers/all-MiniLM-L6-v2", "input": ["Hello world"]}'
```

## Pipeline Integration

This component is conditionally invoked in the pipeline when `deploy_embedding=True`:

```python
with dsl.If(deploy_embedding == True):
    embed_deploy_task = deploy_embedding_model(
        model_name=embedding_model,
        namespace=namespace,
    )
    embed_deploy_task.after(chunk_task)
```

When `deploy_embedding=False`, the pipeline skips this step entirely. The downstream `ingest_to_milvus` component determines embedding mode based on whether `embedding_endpoint` is empty (local mode) or a URL (service mode).

## Metadata

- **Name**: deploy_embedding_model
- **Stability**: experimental
- **Dependencies**:
  - Kubeflow:
    - Name: Pipelines, Version: >=2.15.2
  - External Services:
    - Name: OpenShift AI (KServe), Version: >=2.10
- **Tags**: embeddings, model_serving, kserve, rag
- **Last Verified**: 2026-04-28
- **Owners**:
  - Approvers: szaher

## Dependencies

- **Base image**: `registry.redhat.io/rhoai/odh-pipeline-runtime-datascience-cpu-py312-rhel9`
- **Python packages**: `kubernetes>=28.1.0`
- **Cluster requirements**: KServe with `ServingRuntime` and `InferenceService` CRDs installed
- **Network**: TEI container must be able to pull the model from HuggingFace at startup

## Additional Resources

- **Documentation**: [https://github.com/kubeflow/pipelines-components](https://github.com/kubeflow/pipelines-components)
