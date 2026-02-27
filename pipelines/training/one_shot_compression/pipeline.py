import kfp
import kfp.kubernetes
from kfp import dsl

from components.deployment.kubeflow_model_registry import (
    kubeflow_model_registry as model_registry,
)
from components.training.one_shot_model_compressor import (
    one_shot_model_compressor as llm_compressor,
)

# =============================================================================
# PVC Configuration (COMPILE-TIME settings)
# =============================================================================
PIPELINE_NAME = "one-shot-compression"
PVC_SIZE = "100Gi"
PVC_STORAGE_CLASS = "nfs-csi"
PVC_ACCESS_MODES = ["ReadWriteMany"]
# =============================================================================


@dsl.pipeline(
    name=PIPELINE_NAME,
    description="Pipeline to compress a model with LLMCompressor",
    pipeline_config=dsl.PipelineConfig(
        workspace=dsl.WorkspaceConfig(
            size=PVC_SIZE,
            kubernetes=dsl.KubernetesWorkspaceConfig(
                pvcSpecPatch={
                    "accessModes": PVC_ACCESS_MODES,
                    "storageClassName": PVC_STORAGE_CLASS,
                }
            ),
        ),
    ),
)
def one_shot_compression(
    model_id: str = "ibm-granite/granite-3.3-8b-instruct",
    dataset_id: str = "HuggingFaceH4/ultrachat_200k",
    dataset_split: str = "train_sft",
    quantization_scheme: str = "W4A16",
    quantization_ignore_list: list[str] = ["lm_head"],
    cpu_requests: str = "3000m",
    memory_request: str = "75G",
    accelerator_type: str = "nvidia.com/gpu",
    accelerator_limit: str = "1",
    model_registry_address: str = "",
    model_registry_author: str = "pipeline",
    model_registry_model_name: str = "compressed-model",
    model_registry_model_version: str = "1.0.0",
    model_registry_opt_description: str = "Compressed Model via one shot compression",
    model_registry_opt_format_name: str = "pytorch",
    model_registry_opt_port: int = 8080,
):
    """Compress a model with LLMCompressor and register it in the model registry.

    Runs one-shot quantization on a Hugging Face causal LM using the specified
    calibration dataset and quantization scheme, then registers the compressed
    model artifact in a Kubeflow Model Registry instance.

    Args:
        model_id: Hugging Face model identifier to compress.
        dataset_id: Hugging Face dataset identifier for calibration.
        dataset_split: Dataset split to use for calibration.
        quantization_scheme: Quantization scheme to apply (e.g. "W4A16").
        quantization_ignore_list: Layer names to exclude from quantization.
        cpu_requests: CPU resource request for the compression task.
        memory_request: Memory resource request for the compression task.
        accelerator_type: Accelerator resource type (e.g. "nvidia.com/gpu").
        accelerator_limit: Number of accelerator devices to request.
        model_registry_address: Address of the Model Registry server.
        model_registry_author: Author name for the registered model.
        model_registry_model_name: Name to register the compressed model under.
        model_registry_model_version: Version string for the registered model.
        model_registry_opt_description: Description for the registered model.
        model_registry_opt_format_name: Serialization format of the model.
        model_registry_opt_port: Port of the Model Registry server.
    """
    compression_task = (
        llm_compressor(
            model_id=model_id,
            quantization_scheme=quantization_scheme,
            quantization_ignore_list=quantization_ignore_list,
            dataset_id=dataset_id,
            dataset_split=dataset_split,
        )
        .set_accelerator_type(accelerator_type)
        .set_accelerator_limit(accelerator_limit)
        .set_cpu_request(cpu_requests)
        .set_memory_request(memory_request)
    )

    kfp.kubernetes.use_secret_as_env(
        compression_task,
        secret_name="hf-hub-secret",
        secret_key_to_env={"HF_TOKEN": "HF_TOKEN"},
    )

    # =========================================================================
    # Stage 4: Model Registry
    # =========================================================================
    model_registry_task = model_registry(
        pvc_mount_path=dsl.WORKSPACE_PATH_PLACEHOLDER,
        input_model=compression_task.outputs["output_model"],
        registry_address=model_registry_address,
        model_name=model_registry_model_name,
        model_version=model_registry_model_version,
        registry_port=model_registry_opt_port,
        model_format_name=model_registry_opt_format_name,
        model_description=model_registry_opt_description,
        author=model_registry_author,
        shared_log_file="pipeline_log.txt",
        source_pipeline_name=PIPELINE_NAME,
        source_pipeline_run_id=dsl.PIPELINE_JOB_ID_PLACEHOLDER,
        source_pipeline_run_name=dsl.PIPELINE_JOB_NAME_PLACEHOLDER,
        source_namespace="",
    )
    model_registry_task.set_caching_options(False)
    kfp.kubernetes.set_image_pull_policy(model_registry_task, "IfNotPresent")


if __name__ == "__main__":
    from kfp.compiler import Compiler

    Compiler().compile(
        one_shot_compression,
        package_path=__file__.replace(".py", "_pipeline.yaml"),
    )
