"""Eval Hub Evaluation Component — KServe variant.

Evaluates a fine-tuned model using the Eval Hub service. Creates a KServe
ServingRuntime + InferenceService (matching the RHOAI dashboard pattern) to
serve the fine-tuned model from the workspace PVC, submits benchmarks to
Eval Hub, and polls for results. Both resources are cleaned up after evaluation.
"""

from kfp import dsl

RHOAI_VLLM_IMAGE = (
    "registry.redhat.io/rhaii/vllm-cuda-rhel9@sha256:ad06abf3bb5235ebb5b2df84cd1b9fd09e823f0ff2eebfc82bb4590275ccfe0b"
)


@dsl.component(
    base_image="registry.access.redhat.com/ubi9/python-311:latest",
    packages_to_install=[
        "requests",
    ],
    pip_index_urls=["https://pypi.org/simple"],
)
def evalhub_evaluator_kserve(
    output_metrics: dsl.Output[dsl.Metrics],
    output_results: dsl.Output[dsl.Artifact],
    # --- Eval Hub connection ---
    evalhub_url: str,
    # --- Benchmarks (one of these must be provided) ---
    benchmarks: list = [],
    collection_id: str = "",
    # --- Optional ---
    pvc_mount_path: str = "",
    model_artifact: dsl.Input[dsl.Model] = None,
    model_path: str = None,
    evalhub_tenant: str = "",
    evalhub_auth_token: str = "",
    evalhub_model_name: str = "finetuned-model",
    base_model_name: str = "",
    evalhub_job_name: str = "pipeline-eval",
    evalhub_timeout: int = 7200,
    evalhub_poll_interval: int = 30,
    mlflow_experiment_name: str = "",
    # --- KServe model serving ---
    gpu_count: int = 1,
    memory: str = "8Gi",
    cpu: str = "2",
    runtime_image: str = (  # noqa: E501
        "registry.redhat.io/rhaii/vllm-cuda-rhel9@sha256:ad06abf3bb5235ebb5b2df84cd1b9fd09e823f0ff2eebfc82bb4590275ccfe0b"
    ),
    trust_remote_code: bool = False,
    verify_tls: bool = False,
    isvc_ready_timeout: int = 600,
):
    """Evaluate a model via Eval Hub with a KServe InferenceService.

    Creates a KServe ServingRuntime + InferenceService (matching the RHOAI
    dashboard deployment pattern) to serve the fine-tuned model from the
    workspace PVC. The InferenceService URL is submitted to Eval Hub for
    benchmark evaluation. Both resources are cleaned up after completion.

    Args:
        output_metrics: KFP Metrics artifact for evaluation scores.
        output_results: KFP Artifact for full evaluation results JSON.
        evalhub_url: Eval Hub API endpoint (empty = skip evaluation).
        benchmarks: List of benchmark specs [{"provider_id": "...", "id": "..."}].
        collection_id: Eval Hub collection ID (alternative to benchmarks list).
        pvc_mount_path: Workspace PVC mount path (triggers KFP PVC mount).
        model_artifact: Model artifact from upstream training step.
        model_path: Local filesystem path to model directory (if no artifact).
        evalhub_tenant: Eval Hub tenant / namespace header (X-Tenant).
        evalhub_auth_token: Bearer token for Eval Hub auth.
        evalhub_model_name: Display name for the model in Eval Hub.
        base_model_name: HF model ID for tokenizer resolution.
        evalhub_job_name: Evaluation job name in Eval Hub.
        evalhub_timeout: Max seconds to wait for evaluation to complete.
        evalhub_poll_interval: Seconds between eval status polls.
        mlflow_experiment_name: MLflow experiment name (non-empty enables MLflow).
        gpu_count: Number of GPUs for the InferenceService predictor.
        memory: Pod memory request/limit for the predictor (e.g. "8Gi", "32Gi").
        cpu: CPU request/limit for the predictor (e.g. "2").
        runtime_image: Container image for the ServingRuntime (RHOAI vLLM default).
        trust_remote_code: Pass --trust-remote-code to vLLM (enables arbitrary code from model repos).
        verify_tls: Verify TLS certificates for Eval Hub API calls (False for self-signed certs).
        isvc_ready_timeout: Max seconds to wait for InferenceService readiness.
    """
    import json
    import logging
    import os
    import time
    import uuid

    import requests

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger("evalhub-kserve")

    # =========================================================================
    # Kubernetes API helpers (shared with standalone variant)
    # =========================================================================
    K8S_API = "https://kubernetes.default.svc"
    SA_TOKEN_PATH = "/var/run/secrets/kubernetes.io/serviceaccount/token"
    SA_CA_PATH = "/var/run/secrets/kubernetes.io/serviceaccount/ca.crt"
    SA_NS_PATH = "/var/run/secrets/kubernetes.io/serviceaccount/namespace"

    def _read_sa_token():
        with open(SA_TOKEN_PATH) as f:
            return f.read().strip()

    def _read_namespace():
        with open(SA_NS_PATH) as f:
            return f.read().strip()

    def _k8s_api(method, path, body=None):
        url = f"{K8S_API}{path}"
        headers = {
            "Authorization": f"Bearer {_read_sa_token()}",
            "Content-Type": "application/json",
        }
        resp = requests.request(
            method,
            url,
            headers=headers,
            json=body,
            verify=SA_CA_PATH,
            timeout=30,
        )
        if resp.status_code >= 400:
            logger.error(f"K8s API {method} {path} -> {resp.status_code}: {resp.text[:500]}")
        return resp

    def _get_own_pod(namespace):
        hostname = os.environ.get("HOSTNAME", "")
        if not hostname:
            import socket

            hostname = socket.gethostname()
        resp = _k8s_api("GET", f"/api/v1/namespaces/{namespace}/pods/{hostname}")
        if resp.status_code == 200:
            return resp.json()
        # HOSTNAME is truncated at 63 chars (DNS label limit). List pods and match by prefix.
        logger.info(f"Pod {hostname} not found directly (name truncated?), searching by prefix...")
        resp = _k8s_api("GET", f"/api/v1/namespaces/{namespace}/pods?limit=500")
        resp.raise_for_status()
        for pod in resp.json().get("items", []):
            pod_name = pod["metadata"]["name"]
            if pod_name.startswith(hostname):
                logger.info(f"Found pod by prefix match: {pod_name}")
                return pod
        raise RuntimeError(f"Could not find own pod. Hostname={hostname}")

    def _find_workspace_pvc(pod_spec, model_path):
        volumes = pod_spec.get("spec", {}).get("volumes", [])
        pvc_volumes = {}
        for v in volumes:
            pvc = v.get("persistentVolumeClaim", {})
            if pvc.get("claimName"):
                pvc_volumes[v["name"]] = pvc["claimName"]

        containers = pod_spec.get("spec", {}).get("containers", [])
        for c in containers:
            for vm in c.get("volumeMounts", []):
                vol_name = vm["name"]
                mount_path = vm["mountPath"]
                normalized_mount = mount_path.rstrip("/") + "/"
                if vol_name in pvc_volumes and (model_path + "/").startswith(normalized_mount):
                    return pvc_volumes[vol_name], mount_path

        raise RuntimeError(f"Could not find workspace PVC for model path {model_path}. PVC volumes: {pvc_volumes}")

    # =========================================================================
    # KServe resource helpers
    # =========================================================================
    KSERVE_SR_API = "/apis/serving.kserve.io/v1alpha1"
    KSERVE_ISVC_API = "/apis/serving.kserve.io/v1beta1"

    def _create_serving_runtime(namespace, name, image, served_model_name, enable_trust_remote_code=False):
        vllm_args = [
            "--port=8080",
            "--model=/mnt/models",
            f"--served-model-name={served_model_name}",
        ]
        if enable_trust_remote_code:
            vllm_args.append("--trust-remote-code")
        sr = {
            "apiVersion": "serving.kserve.io/v1alpha1",
            "kind": "ServingRuntime",
            "metadata": {
                "name": name,
                "namespace": namespace,
                "annotations": {
                    "opendatahub.io/apiProtocol": "REST",
                    "opendatahub.io/template-name": "vllm-cuda-runtime-template",
                    "opendatahub.io/template-display-name": "vLLM NVIDIA GPU ServingRuntime for KServe",
                    "openshift.io/display-name": f"EvalHub vLLM Runtime {name}",
                },
                "labels": {
                    "opendatahub.io/dashboard": "true",
                },
            },
            "spec": {
                "annotations": {
                    "prometheus.io/path": "/metrics",
                    "prometheus.io/port": "8080",
                },
                "containers": [
                    {
                        "name": "kserve-container",
                        "image": image,
                        "command": ["python", "-m", "vllm.entrypoints.openai.api_server"],
                        "args": vllm_args,
                        "env": [
                            {"name": "HF_HOME", "value": "/tmp/hf_home"},
                        ],
                        "ports": [{"containerPort": 8080, "protocol": "TCP"}],
                    }
                ],
                "multiModel": False,
                "supportedModelFormats": [
                    {"autoSelect": True, "name": "vLLM"},
                ],
            },
        }
        path = f"{KSERVE_SR_API}/namespaces/{namespace}/servingruntimes"
        resp = _k8s_api("POST", path, sr)
        resp.raise_for_status()
        logger.info(f"Created ServingRuntime {name}")
        return resp.json()

    def _create_inference_service(namespace, name, runtime_name, pvc_name, model_relative_path, n_gpu, mem, n_cpu):
        storage_uri = f"pvc://{pvc_name}/{model_relative_path}"

        isvc = {
            "apiVersion": "serving.kserve.io/v1beta1",
            "kind": "InferenceService",
            "metadata": {
                "name": name,
                "namespace": namespace,
                "annotations": {
                    "serving.kserve.io/deploymentMode": "Standard",
                    "security.opendatahub.io/enable-auth": "false",
                    "openshift.io/display-name": f"EvalHub Model {name}",
                },
                "labels": {
                    "opendatahub.io/dashboard": "true",
                },
            },
            "spec": {
                "predictor": {
                    "maxReplicas": 1,
                    "minReplicas": 1,
                    "model": {
                        "modelFormat": {"name": "vLLM"},
                        "runtime": runtime_name,
                        "storageUri": storage_uri,
                        "resources": {
                            "limits": {
                                "nvidia.com/gpu": str(n_gpu),
                                "cpu": n_cpu,
                                "memory": mem,
                            },
                            "requests": {
                                "nvidia.com/gpu": str(n_gpu),
                                "cpu": n_cpu,
                                "memory": mem,
                            },
                        },
                    },
                    "timeout": 30,
                },
            },
        }
        path = f"{KSERVE_ISVC_API}/namespaces/{namespace}/inferenceservices"
        resp = _k8s_api("POST", path, isvc)
        resp.raise_for_status()
        logger.info(f"Created InferenceService {name} (storageUri={storage_uri})")
        return resp.json()

    def _wait_for_isvc_ready(namespace, name, timeout_s=600):
        start = time.time()
        path = f"{KSERVE_ISVC_API}/namespaces/{namespace}/inferenceservices/{name}"
        logger.info(f"Waiting for InferenceService {name} to be Ready (timeout: {timeout_s}s)...")
        while time.time() - start < timeout_s:
            resp = _k8s_api("GET", path)
            if resp.status_code == 200:
                isvc = resp.json()
                conditions = isvc.get("status", {}).get("conditions", [])
                for c in conditions:
                    if c.get("type") == "Ready" and c.get("status") == "True":
                        duration = time.time() - start
                        logger.info(f"InferenceService {name} ready in {duration:.1f}s")
                        return duration
                reasons = [f"{c['type']}={c.get('status', '?')}({c.get('reason', '')})" for c in conditions]
                logger.info(f"  ISVC conditions: {', '.join(reasons) if reasons else 'none yet'}")
            time.sleep(15)
        raise TimeoutError(f"InferenceService {name} did not become Ready within {timeout_s}s")

    def _get_isvc_url(namespace, name):
        """Get the model URL from the ISVC status.address.url field."""
        path = f"{KSERVE_ISVC_API}/namespaces/{namespace}/inferenceservices/{name}"
        resp = _k8s_api("GET", path)
        if resp.status_code == 200:
            isvc = resp.json()
            address_url = isvc.get("status", {}).get("address", {}).get("url", "")
            if address_url:
                url = address_url.rstrip("/") + "/v1"
                logger.info(f"ISVC address URL: {address_url}")
                logger.info(f"Model URL: {url}")
                return url
            status_url = isvc.get("status", {}).get("url", "")
            if status_url:
                url = status_url.rstrip("/") + ":8080/v1"
                logger.info(f"ISVC status URL: {status_url}, using port 8080")
                logger.info(f"Model URL: {url}")
                return url
        fallback = f"http://{name}-predictor.{namespace}.svc.cluster.local:8080/v1"
        logger.warning(f"Could not read ISVC status, falling back to: {fallback}")
        return fallback

    def _cleanup_kserve(namespace, sr_name, isvc_name):
        logger.info(f"Cleaning up InferenceService {isvc_name} and ServingRuntime {sr_name}")
        for kind, api, name in [
            ("InferenceService", KSERVE_ISVC_API, isvc_name),
            ("ServingRuntime", KSERVE_SR_API, sr_name),
        ]:
            resp = _k8s_api("DELETE", f"{api}/namespaces/{namespace}/{kind.lower()}s/{name}")
            if resp.status_code >= 400 and resp.status_code != 404:
                logger.warning(f"Failed to delete {kind} {name}: {resp.status_code} {resp.text[:200]}")
            else:
                logger.info(f"Deleted {kind} {name}")
        logger.info(f"Cleanup complete for {isvc_name} / {sr_name}")

    # =========================================================================
    # 0. Check if evaluation should run
    # =========================================================================
    if not evalhub_url or not evalhub_url.strip():
        logger.info("Eval Hub URL not provided — skipping evaluation.")
        output_metrics.metadata["evalhub_state"] = "skipped"
        output_metrics.metadata["evalhub_url"] = ""
        results_path = os.path.join(output_results.path, "eval_results.json")
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        with open(results_path, "w") as f:
            json.dump({"state": "skipped", "reason": "evalhub_url not provided"}, f)
        output_results.metadata["evalhub_state"] = "skipped"
        return

    # =========================================================================
    # 1. Resolve model path
    # =========================================================================
    final_model_path = None

    if model_artifact is not None:
        pvc_dir = model_artifact.metadata.get("pvc_model_dir", "")
        if pvc_dir and os.path.isdir(pvc_dir):
            final_model_path = pvc_dir
            logger.info(f"Using model from PVC: {final_model_path}")
        elif os.path.isdir(model_artifact.path):
            final_model_path = model_artifact.path
            logger.info(f"Using model from artifact path: {final_model_path}")

    if final_model_path is None and model_path:
        final_model_path = model_path
        logger.info(f"Using model_path: {final_model_path}")

    if final_model_path is None:
        raise ValueError("No model provided. Supply model_artifact or model_path.")

    config_json = os.path.join(final_model_path, "config.json")
    if os.path.isdir(final_model_path) and not os.path.exists(config_json):
        raise FileNotFoundError(f"Model directory exists but missing config.json: {final_model_path}")

    # =========================================================================
    # 2. Deploy model via KServe (ServingRuntime + InferenceService)
    # =========================================================================
    resolved_model_name = base_model_name if base_model_name else evalhub_model_name
    namespace = _read_namespace()
    short_id = uuid.uuid4().hex[:8]
    sr_name = f"evalhub-rt-{short_id}"
    isvc_name = f"evalhub-isvc-{short_id}"

    logger.info(f"Namespace: {namespace}")
    logger.info(f"ServingRuntime: {sr_name}")
    logger.info(f"InferenceService: {isvc_name}")
    logger.info(f"Model path: {final_model_path}")
    logger.info(f"Runtime image: {runtime_image}")
    logger.info(f"Resources: {gpu_count} GPU(s), {cpu} CPU, {memory} memory")

    pod_spec = _get_own_pod(namespace)
    workspace_pvc_name, workspace_mount = _find_workspace_pvc(pod_spec, final_model_path)
    logger.info(f"Workspace PVC: {workspace_pvc_name} mounted at {workspace_mount}")

    model_relative_path = final_model_path
    if final_model_path.startswith(workspace_mount):
        model_relative_path = final_model_path[len(workspace_mount) :].lstrip("/")
    logger.info(f"Model relative path in PVC: {model_relative_path}")
    logger.info(f"storageUri will be: pvc://{workspace_pvc_name}/{model_relative_path}")

    _create_serving_runtime(namespace, sr_name, runtime_image, resolved_model_name, trust_remote_code)

    try:
        _create_inference_service(
            namespace=namespace,
            name=isvc_name,
            runtime_name=sr_name,
            pvc_name=workspace_pvc_name,
            model_relative_path=model_relative_path,
            n_gpu=gpu_count,
            mem=memory,
            n_cpu=cpu,
        )

        isvc_startup_duration = _wait_for_isvc_ready(namespace, isvc_name, timeout_s=isvc_ready_timeout)

        model_url = _get_isvc_url(namespace, isvc_name)
        logger.info(f"KServe model serving at {model_url}")

        # =====================================================================
        # 3. Submit evaluation job to Eval Hub
        # =====================================================================
        headers = {"Content-Type": "application/json", "Accept": "application/json"}

        auth_token = evalhub_auth_token
        if not auth_token:
            if os.path.exists(SA_TOKEN_PATH):
                auth_token = _read_sa_token()
                logger.info("Using pod ServiceAccount token for Eval Hub auth")
        if auth_token:
            headers["Authorization"] = f"Bearer {auth_token}"

        tenant = evalhub_tenant
        if not tenant:
            tenant = namespace
            logger.info(f"Using namespace as tenant: {tenant}")
        headers["X-Tenant"] = tenant

        model_spec = {
            "url": model_url,
            "name": resolved_model_name,
        }
        logger.info(f"Model name for Eval Hub: {resolved_model_name}")

        eval_config = {
            "name": evalhub_job_name,
            "model": model_spec,
        }
        if mlflow_experiment_name.strip():
            eval_config["experiment"] = {"name": mlflow_experiment_name.strip()}
            logger.info(f"MLflow enabled (experiment: {mlflow_experiment_name.strip()})")
        else:
            logger.info("MLflow disabled (no mlflow_experiment_name provided)")

        cleaned_collection = collection_id.strip().strip('"').strip("'") if collection_id else ""
        if cleaned_collection:
            eval_config["collection"] = {"id": cleaned_collection}
        elif benchmarks:
            parsed = benchmarks if isinstance(benchmarks, list) else json.loads(benchmarks)
            eval_config["benchmarks"] = parsed
        else:
            raise ValueError("Provide either 'benchmarks' list or 'collection_id'.")

        submit_url = f"{evalhub_url.rstrip('/')}/api/v1/evaluations/jobs"
        logger.info(f"Submitting evaluation job to {submit_url}")
        logger.info(f"Config: {json.dumps(eval_config, indent=2)}")

        resp = requests.post(submit_url, json=eval_config, headers=headers, timeout=30, verify=verify_tls)
        if resp.status_code not in (200, 201, 202):
            raise RuntimeError(f"Eval Hub returned {resp.status_code}: {resp.text}")

        job = resp.json()
        job_id = job["resource"]["id"]
        logger.info(f"Evaluation job created: {job_id}")

        # =====================================================================
        # 4. Poll for completion
        # =====================================================================
        job_url = f"{evalhub_url.rstrip('/')}/api/v1/evaluations/jobs/{job_id}"
        terminal_states = {"completed", "failed", "cancelled", "partially_failed"}
        poll_start = time.time()

        while time.time() - poll_start < evalhub_timeout:
            time.sleep(evalhub_poll_interval)

            try:
                resp = requests.get(job_url, headers=headers, timeout=30, verify=verify_tls)
                if resp.status_code != 200:
                    logger.warning(f"Poll returned {resp.status_code}, retrying...")
                    continue

                job = resp.json()
                state = job.get("status", {}).get("state", "unknown")
                benchmark_statuses = job.get("status", {}).get("benchmarks", [])
                completed_count = sum(1 for b in benchmark_statuses if b.get("status") in terminal_states)
                total_count = len(benchmark_statuses)

                logger.info(f"Job {job_id}: state={state}, benchmarks={completed_count}/{total_count}")

                if state in terminal_states:
                    logger.info(f"Evaluation reached terminal state: {state}")
                    break
            except requests.RequestException as e:
                logger.warning(f"Poll error: {e}, retrying...")
        else:
            logger.error(f"Evaluation timed out after {evalhub_timeout}s")
            try:
                cancel_url = f"{evalhub_url.rstrip('/')}/api/v1/evaluations/jobs/{job_id}"
                requests.delete(cancel_url, headers=headers, timeout=10, verify=verify_tls)
                logger.info(f"Cancelled job {job_id}")
            except Exception:
                pass
            raise TimeoutError(f"Evaluation did not complete within {evalhub_timeout}s")

        # =====================================================================
        # 5. Process results
        # =====================================================================
        final_state = job.get("status", {}).get("state", "unknown")
        results = job.get("results", {})
        mlflow_experiment_url = results.get("mlflow_experiment_url", "")
        benchmark_results = results.get("benchmarks", [])
        test_result = results.get("test", {})

        logger.info("=" * 60)
        logger.info("EVALUATION RESULTS")
        logger.info("=" * 60)
        logger.info(f"  State: {final_state}")
        if mlflow_experiment_url:
            logger.info(f"  MLflow: {mlflow_experiment_url}")
        if test_result:
            logger.info(f"  Overall pass: {test_result.get('pass', 'N/A')}")
            logger.info(f"  Overall score: {test_result.get('score', 'N/A')}")

        output_metrics.metadata["evalhub_job_id"] = job_id
        output_metrics.metadata["evalhub_state"] = final_state
        output_metrics.metadata["evalhub_url"] = evalhub_url
        output_metrics.metadata["model_name"] = evalhub_model_name
        output_metrics.metadata["model_path"] = str(final_model_path)
        output_metrics.metadata["serving_method"] = "kserve"
        output_metrics.metadata["isvc_name"] = isvc_name
        output_metrics.metadata["isvc_startup_seconds"] = round(isvc_startup_duration, 1)
        if mlflow_experiment_name.strip():
            output_metrics.metadata["mlflow_experiment_name"] = mlflow_experiment_name.strip()

        if mlflow_experiment_url:
            output_metrics.metadata["mlflow_experiment_url"] = mlflow_experiment_url

        if test_result:
            if "pass" in test_result:
                output_metrics.log_metric("eval_pass", 1.0 if test_result["pass"] else 0.0)
            if "score" in test_result and test_result["score"] is not None:
                output_metrics.log_metric("eval_overall_score", float(test_result["score"]))

        for br in benchmark_results:
            bench_id = br.get("id", "unknown")
            provider_id = br.get("provider_id", "unknown")
            prefix = f"{provider_id}/{bench_id}"
            metrics = br.get("metrics", {})
            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, (int, float)):
                    output_metrics.log_metric(f"{prefix}/{metric_name}", float(metric_value))
                    logger.info(f"  {prefix}/{metric_name}: {metric_value}")

            bench_test = br.get("test", {})
            if bench_test and "pass" in bench_test:
                output_metrics.log_metric(f"{prefix}/pass", 1.0 if bench_test["pass"] else 0.0)

        results_data = {
            "job_id": job_id,
            "state": final_state,
            "mlflow_experiment_url": mlflow_experiment_url,
            "results": results,
            "config": eval_config,
            "model_url": model_url,
            "serving_method": "kserve",
            "isvc_name": isvc_name,
            "isvc_startup_seconds": round(isvc_startup_duration, 1),
        }

        results_path = os.path.join(output_results.path, "eval_results.json")
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        with open(results_path, "w") as f:
            json.dump(results_data, f, indent=2, default=str)
        logger.info(f"Results written to {results_path}")

        output_results.metadata["evalhub_job_id"] = job_id
        output_results.metadata["evalhub_state"] = final_state
        if mlflow_experiment_url:
            output_results.metadata["mlflow_experiment_url"] = mlflow_experiment_url

        if final_state == "failed":
            msg = job.get("status", {}).get("message", {})
            error_text = msg.get("text", "Unknown error") if isinstance(msg, dict) else str(msg)
            raise RuntimeError(f"Evaluation failed: {error_text}")

    finally:
        _cleanup_kserve(namespace, sr_name, isvc_name)

    logger.info("Eval Hub evaluation (KServe) complete.")
