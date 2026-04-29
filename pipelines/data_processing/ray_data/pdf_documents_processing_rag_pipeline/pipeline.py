"""KFP Pipeline: Multi-Step RAG Document Processing.

Orchestrates up to five reusable components:
1. Parse & chunk PDFs (Docling + HybridChunker via RayJob -> S3)
2. Deploy embedding model (optional — only when deploy_embedding=True)
3. Ingest into Milvus (read chunks from S3, embed locally or via service, insert)
4. Download LLM to PVC (cached — skips if already present)
5. Deploy LLM for RAG inference (vLLM InferenceService from PVC)

Intermediate chunks are stored in S3 (MinIO), not on the PVC.
The data PVC is mounted read-only for input PDFs.
The model cache PVC stores downloaded HuggingFace models.
Embedding supports two modes: local sentence-transformers or a deployed service.
S3 credentials are read from a Kubernetes Secret (not pipeline parameters).
"""

from kfp import dsl
from kfp import kubernetes

from kfp_components.components.data_processing.download_model import download_model
from kfp_components.components.data_processing.ingest_to_milvus import ingest_to_milvus
from kfp_components.components.data_processing.parse_and_chunk import parse_and_chunk
from kfp_components.components.deployment.deploy_embedding_model import deploy_embedding_model
from kfp_components.components.deployment.model_deployment import model_deployment


@dsl.pipeline(
    name="RAG Multi-Step Pipeline",
    description=(
        "Multi-step RAG pipeline: parse & chunk PDFs with Docling "
        "(output to S3), ingest into Milvus with local or deployed embeddings, "
        "optionally deploy an embedding service, "
        "download and deploy an LLM for inference (cached on PVC)."
    ),
)
def rag_multistep_pipeline(
    # Shared
    pvc_name: str = "data-pvc",
    pvc_mount_path: str = "/mnt/data",
    namespace: str = "ray-docling",
    # S3 (MinIO)
    s3_endpoint: str = "http://minio-service.default.svc.cluster.local:9000",
    s3_bucket: str = "rag-chunks",
    s3_prefix: str = "chunks",
    s3_secret_name: str = "minio-secret",
    # PDF parsing
    input_path: str = "input/pdfs",
    ray_image: str = "quay.io/rhoai-szaher/docling-ray:latest",
    num_workers: int = 2,
    worker_cpus: int = 8,
    worker_memory_gb: int = 16,
    head_cpus: int = 2,
    head_memory_gb: int = 8,
    cpus_per_actor: int = 4,
    min_actors: int = 2,
    max_actors: int = 4,
    batch_size: int = 4,
    chunk_max_tokens: int = 256,
    num_files: int = 1000,
    timeout_seconds: int = 600,
    enable_profiling: bool = False,
    verbose: bool = True,
    # Embedding
    deploy_embedding: bool = False,  # If True, deploys embedding model as InferenceService
    embedding_endpoint: str = "",  # If empty, uses local model; else uses deployed service
    embedding_model: str = "ibm-granite/granite-embedding-125m-english",
    embedding_dim: int = 768,
    embedding_runtime_image: str = "registry.redhat.io/rhaiis/vllm-cuda-rhel9@sha256:094db84a1da5e8a575d0c9eade114fa30f4a2061064a338e3e032f3578f8082a",
    embedding_gpu_count: int = 1,
    # Milvus
    milvus_host: str = "milvus-milvus.milvus.svc.cluster.local",
    milvus_port: int = 19530,
    milvus_db: str = "default",
    collection_name: str = "rag_documents",
    embed_batch_size: int = 64,
    milvus_batch_size: int = 256,
    # LLM deployment
    llm_model_name: str = "mistralai/Mistral-7B-Instruct-v0.3",
    model_cache_pvc: str = "model-cache-pvc",
    max_model_len: int = 4096,
    gpu_count: int = 1,
):
    # Step 1: Parse & chunk PDFs -> S3
    chunk_task = parse_and_chunk(
        pvc_name=pvc_name,
        pvc_mount_path=pvc_mount_path,
        input_path=input_path,
        ray_image=ray_image,
        namespace=namespace,
        s3_endpoint=s3_endpoint,
        s3_bucket=s3_bucket,
        s3_prefix=s3_prefix,
        tokenizer=embedding_model,
        chunk_max_tokens=chunk_max_tokens,
        num_workers=num_workers,
        worker_cpus=worker_cpus,
        worker_memory_gb=worker_memory_gb,
        head_cpus=head_cpus,
        head_memory_gb=head_memory_gb,
        cpus_per_actor=cpus_per_actor,
        min_actors=min_actors,
        max_actors=max_actors,
        batch_size=batch_size,
        num_files=num_files,
        timeout_seconds=timeout_seconds,
        enable_profiling=enable_profiling,
        verbose=verbose,
    )
    kubernetes.use_secret_as_env(
        chunk_task,
        secret_name=s3_secret_name,
        secret_key_to_env={
            "access_key": "S3_ACCESS_KEY",
            "secret_key": "S3_SECRET_KEY",
        },
    )
    chunk_task.set_caching_options(False)

    # Step 2 (optional): Deploy embedding model as InferenceService
    # When deploy_embedding=True, the pipeline deploys the embedding
    # model and passes its endpoint to the ingest step.
    # When deploy_embedding=False, embedding_endpoint is used as-is
    # (empty string = local model, URL = existing service).
    with dsl.If(deploy_embedding == True):  # noqa: E712
        embed_deploy_task = deploy_embedding_model(
            model_name=embedding_model,
            namespace=namespace,
            runtime_image=embedding_runtime_image,
            gpu_count=embedding_gpu_count,
        )
        embed_deploy_task.after(chunk_task)
        embed_deploy_task.set_caching_options(False)

    # Step 3: Ingest into Milvus (supports local or deployed embedding model)
    ingest_task = ingest_to_milvus(
        s3_endpoint=s3_endpoint,
        s3_bucket=s3_bucket,
        s3_prefix=s3_prefix,
        milvus_host=milvus_host,
        milvus_port=milvus_port,
        milvus_db=milvus_db,
        collection_name=collection_name,
        embedding_endpoint=embedding_endpoint,
        embedding_model=embedding_model,
        embedding_dim=embedding_dim,
        embed_batch_size=embed_batch_size,
        milvus_batch_size=milvus_batch_size,
    )
    kubernetes.use_secret_as_env(
        ingest_task,
        secret_name=s3_secret_name,
        secret_key_to_env={
            "access_key": "S3_ACCESS_KEY",
            "secret_key": "S3_SECRET_KEY",
        },
    )
    ingest_task.after(chunk_task)
    ingest_task.set_caching_options(False)

    # Step 4: Download LLM to PVC (skips if already cached)
    download_task = download_model(
        model_name=llm_model_name,
        model_cache_pvc=model_cache_pvc,
    )
    kubernetes.mount_pvc(
        download_task,
        pvc_name=model_cache_pvc,
        mount_path="/mnt/models",
    )

    # Step 5: Deploy LLM for RAG inference (from PVC cache)
    deploy_task = model_deployment(
        model_name=llm_model_name,
        namespace=namespace,
        model_dir=download_task.output,
        model_cache_pvc=model_cache_pvc,
        max_model_len=max_model_len,
        gpu_count=gpu_count,
    )
    deploy_task.set_caching_options(False)
    # No dependency on ingest_task — model deployment runs in parallel
    # with the data pipeline (parse → ingest). Both chains are independent:
    #   Data chain:  parse_and_chunk → ingest_to_milvus
    #   Model chain: download_model  → model_deployment


if __name__ == "__main__":
    from kfp import compiler

    compiler.Compiler().compile(
        rag_multistep_pipeline,
        package_path="rag_multistep_pipeline.yaml",
    )
    print("Pipeline compiled to rag_multistep_pipeline.yaml")
