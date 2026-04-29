# Model Deployment Component

> **Stability: experimental** — This asset is not yet stable and may change.

KFP component that deploys an LLM on OpenShift AI using vLLM. Creates a KServe ServingRuntime and InferenceService matching the RHOAI dashboard deployment pattern, serving the model from a PVC cache with GPU acceleration.

## How It Works

1. **Derive Names** -- Converts the HuggingFace model ID to a Kubernetes-safe name (e.g. `mistralai/Mistral-7B-Instruct-v0.3` becomes `mistral-7b-instruct-v0-3`). This name is used for both the ServingRuntime and InferenceService.
2. **Lookup HardwareProfile** -- Attempts to read the specified `HardwareProfile` CR from the configured namespace. If found, its `resourceVersion` is included in InferenceService annotations for RHOAI dashboard compatibility. If not found, proceeds without it.
3. **Create ServingRuntime** -- Creates or updates a `ServingRuntime` CR configured with:
   - Red Hat vLLM CUDA container image
   - OpenAI-compatible API server entrypoint (`vllm.entrypoints.openai.api_server`)
   - Flags: `--max-model-len`, `--enforce-eager`, `--gpu-memory-utilization=0.99`, `--max-num-seqs=16`
   - RHOAI dashboard labels and annotations for UI visibility
4. **Delete Existing InferenceService** -- If an InferenceService with the same name exists, it is deleted first and the component waits up to 2 minutes for full cleanup. This avoids stuck rolling updates when changing model parameters or GPU configuration.
5. **Create InferenceService** -- Creates a fresh `InferenceService` CR with:
   - `RawDeployment` mode (no Knative dependency)
   - `storageUri` pointing to the PVC-cached model (`pvc://{model_cache_pvc}/{model_dir}`)
   - GPU resource requests/limits (`nvidia.com/gpu`)
   - RHOAI dashboard labels (`opendatahub.io/dashboard: true`)
   - Auth disabled for in-cluster access (`security.opendatahub.io/enable-auth: false`)
6. **Wait for Ready** -- Polls the InferenceService status every 30 seconds (up to 30 minutes) until the `Ready` condition is `True`.

### Why Delete-and-Recreate?

Unlike the create-or-update pattern used in other components, this component deletes the existing InferenceService before creating a new one. This is intentional: KServe `RawDeployment` mode can get stuck in a rolling update when GPU resources or model paths change, because the old pod holds the GPU lease until the new pod is ready, creating a deadlock. Deleting first ensures the GPU is released.

## Inputs

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name` | `str` | *required* | HuggingFace model ID (e.g. `mistralai/Mistral-7B-Instruct-v0.3`) |
| `namespace` | `str` | *required* | Namespace to deploy the InferenceService |
| `model_dir` | `str` | *required* | PVC sub-path with model files (output from `download_model`) |
| `model_cache_pvc` | `str` | *required* | Name of the PVC containing cached model weights |
| `hardware_profile_name` | `str` | `"gpu-profile"` | HardwareProfile CR name for GPU resources |
| `hardware_profile_namespace` | `str` | `"redhat-ods-applications"` | Namespace of the HardwareProfile CR |
| `min_replicas` | `int` | `1` | Minimum number of replicas |
| `max_replicas` | `int` | `1` | Maximum number of replicas |
| `gpu_count` | `int` | `1` | Number of NVIDIA GPUs per replica |
| `max_model_len` | `int` | `4096` | Maximum context length (limits KV cache memory) |
| `cpu_requests` | `str` | `"2"` | CPU requests for the predictor pod |
| `memory_requests` | `str` | `"8Gi"` | Memory requests for the predictor pod |
| `cpu_limits` | `str` | `"2"` | CPU limits for the predictor pod |
| `memory_limits` | `str` | `"8Gi"` | Memory limits for the predictor pod |

## Output

| Type | Description |
|------|-------------|
| `str` | Inference endpoint URL with `/v1` suffix (e.g. `http://mistral-7b-instruct-v0-3.ray-docling.svc.cluster.local/v1`) |

The endpoint exposes an OpenAI-compatible API:

```bash
# List models
curl http://<endpoint>/v1/models

# Chat completion
curl http://<endpoint>/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "mistral-7b-instruct-v0-3", "messages": [{"role": "user", "content": "Hello"}]}'
```

## Pipeline Integration

This component is the final step in the model chain. It depends on `download_model` for the `model_dir` output:

```python
download_task = download_model(model_name=llm_model_name, model_cache_pvc=model_cache_pvc)
kubernetes.mount_pvc(download_task, pvc_name=model_cache_pvc, mount_path="/mnt/models")

deploy_task = model_deployment(
    model_name=llm_model_name,
    namespace=namespace,
    model_dir=download_task.output,  # PVC sub-path from download_model
    model_cache_pvc=model_cache_pvc,
)
```

The model chain (download -> deploy) runs in parallel with the data chain (parse -> ingest).

## Metadata

- **Name**: model_deployment
- **Stability**: experimental
- **Dependencies**:
  - Kubeflow:
    - Name: Pipelines, Version: >=2.15.2
  - External Services:
    - Name: KServe, Version: >=0.11.0
    - Name: vLLM ServingRuntime, Version: >=0.4.0
- **Tags**: deployment, model_serving, vllm, kserve, rag
- **Last Verified**: 2026-04-28
- **Owners**:
  - Approvers: szaher

## Dependencies

- **Base image**: `registry.redhat.io/rhoai/odh-pipeline-runtime-datascience-cpu-py312-rhel9`
- **Python packages**: `kubernetes>=28.1.0`
- **Cluster requirements**: KServe CRDs, NVIDIA GPU operator, vLLM CUDA container image
- **Upstream dependency**: Requires `download_model` output as `model_dir` input

## Additional Resources

- **Documentation**: [https://github.com/kubeflow/pipelines-components](https://github.com/kubeflow/pipelines-components)
