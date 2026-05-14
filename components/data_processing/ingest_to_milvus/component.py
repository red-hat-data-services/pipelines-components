"""KFP component: Read chunks from S3, embed, and ingest into Milvus.

Reads JSONL chunk files from an S3-compatible bucket, generates embeddings
using either a deployed embedding service endpoint or a local
sentence-transformers model, and inserts the vectors into Milvus.
"""

from kfp import dsl


@dsl.component(
    base_image="registry.redhat.io/rhoai/odh-pipeline-runtime-datascience-cpu-py312-rhel9@sha256:f9844dc150592a9f196283b3645dda92bd80dfdb3d467fa8725b10267ea5bdbc",
    packages_to_install=[
        "pymilvus>=2.4.0",
        "sentence-transformers>=2.2.0",
        "requests>=2.28.0",
        "boto3>=1.28.0",
    ],
)
def ingest_to_milvus(
    s3_endpoint: str,
    s3_bucket: str,
    milvus_host: str,
    s3_prefix: str = "chunks",
    embedding_endpoint: str = "",
    embedding_model: str = "ibm-granite/granite-embedding-125m-english",
    embedding_dim: int = 768,
    milvus_port: int = 19530,
    milvus_db: str = "default",
    collection_name: str = "rag_documents",
    drop_existing: bool = True,
    embed_batch_size: int = 64,
    milvus_batch_size: int = 256,
) -> str:
    """Read chunks from S3, embed, and insert into Milvus.

    Args:
        s3_endpoint: S3-compatible endpoint URL (e.g. MinIO).
        s3_bucket: S3 bucket containing chunk files.
        milvus_host: Milvus service hostname.
        s3_prefix: Key prefix for chunk files in S3.
        embedding_endpoint: Optional embedding service URL. If empty,
            uses a local sentence-transformers model.
        embedding_model: Embedding model name (for API or local).
        embedding_dim: Dimension of the embedding vectors.
        milvus_port: Milvus gRPC port.
        milvus_db: Milvus database name.
        collection_name: Milvus collection name.
        drop_existing: If True, drop and recreate the collection. If False, append to it.
        embed_batch_size: Batch size for embedding requests.
        milvus_batch_size: Batch size for Milvus inserts.

    Returns:
        The Milvus collection name and total vectors inserted.
    """
    import json
    import os
    import time

    import boto3
    import requests as req_lib
    from pymilvus import CollectionSchema, DataType, FieldSchema, MilvusClient

    # --- Read chunks from S3 ---
    s3 = boto3.client(
        "s3",
        endpoint_url=s3_endpoint,
        aws_access_key_id=os.environ["S3_ACCESS_KEY"],
        aws_secret_access_key=os.environ["S3_SECRET_KEY"],
        region_name="us-east-1",
    )

    all_chunks = []
    file_count = 0
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=s3_bucket, Prefix=s3_prefix + "/"):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith(".jsonl"):
                continue
            resp = s3.get_object(Bucket=s3_bucket, Key=key)
            body = resp["Body"].read().decode("utf-8")
            for line in body.strip().split("\n"):
                if line:
                    all_chunks.append(json.loads(line))
            file_count += 1

    print(f"Read {len(all_chunks)} chunks from {file_count} JSONL files in s3://{s3_bucket}/{s3_prefix}/")

    if not all_chunks:
        raise FileNotFoundError(f"No chunks found in s3://{s3_bucket}/{s3_prefix}/")

    # --- Setup Milvus collection ---
    uri = f"http://{milvus_host}:{milvus_port}"
    client = MilvusClient(uri=uri, db_name=milvus_db)

    collection_exists = client.has_collection(collection_name)

    if collection_exists and not drop_existing:
        desc = client.describe_collection(collection_name)
        for field in desc.get("fields", []):
            if field.get("name") == "embedding":
                existing_dim = field.get("params", {}).get("dim")
                if existing_dim is not None and int(existing_dim) != embedding_dim:
                    raise ValueError(
                        f"Existing collection '{collection_name}' has dim={existing_dim}, "
                        f"but embedding_dim={embedding_dim}. Drop the collection or fix the dimension."
                    )
                break
        print(f"Appending to existing collection '{collection_name}'.")
    else:
        if collection_exists:
            print(f"Dropping existing collection '{collection_name}'")
            client.drop_collection(collection_name)

        schema = CollectionSchema(
            fields=[
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="source_file", dtype=DataType.VARCHAR, max_length=512),
                FieldSchema(name="chunk_index", dtype=DataType.INT64),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=32768),
                FieldSchema(
                    name="embedding",
                    dtype=DataType.FLOAT_VECTOR,
                    dim=embedding_dim,
                ),
            ],
            description="RAG document chunks with embeddings",
        )
        client.create_collection(collection_name=collection_name, schema=schema)

        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            index_type="IVF_FLAT",
            metric_type="COSINE",
            params={"nlist": 128},
        )
        client.create_index(collection_name=collection_name, index_params=index_params)
        print(f"Collection '{collection_name}' created (dim={embedding_dim}).")

    # --- Setup embedding ---
    use_endpoint = bool(embedding_endpoint)
    local_model = None

    if use_endpoint:
        print(f"Using embedding endpoint: {embedding_endpoint}")
    else:
        print(f"Using local embedding model: {embedding_model}")
        from sentence_transformers import SentenceTransformer

        local_model = SentenceTransformer(embedding_model)

    # --- Embed and insert ---
    start_time = time.time()
    total_inserted = 0

    for i in range(0, len(all_chunks), milvus_batch_size):
        batch = all_chunks[i : i + milvus_batch_size]
        texts = [c["text"] for c in batch]

        # Generate embeddings
        if use_endpoint:
            all_embeddings = []
            for j in range(0, len(texts), embed_batch_size):
                embed_batch = texts[j : j + embed_batch_size]
                resp = req_lib.post(
                    f"{embedding_endpoint}/v1/embeddings",
                    json={"model": embedding_model, "input": embed_batch},
                    timeout=120,
                )
                resp.raise_for_status()
                data = resp.json()["data"]
                data.sort(key=lambda x: x["index"])
                all_embeddings.extend([d["embedding"] for d in data])
            embeddings = all_embeddings
        else:
            embeddings = local_model.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=embed_batch_size,
            ).tolist()

        # Insert into Milvus
        data = [
            {
                "source_file": c["source_file"],
                "chunk_index": c["chunk_index"],
                "text": c["text"],
                "embedding": emb,
            }
            for c, emb in zip(batch, embeddings)
        ]
        client.insert(collection_name=collection_name, data=data)
        total_inserted += len(data)

        if (i // milvus_batch_size) % 10 == 0:
            elapsed = time.time() - start_time
            print(f"  Inserted {total_inserted}/{len(all_chunks)} ({elapsed:.1f}s)")

    # Load collection for searching
    client.load_collection(collection_name)
    stats = client.get_collection_stats(collection_name)

    wall_clock = time.time() - start_time
    print(f"\nIngestion complete: {total_inserted} vectors in {wall_clock:.1f}s")
    print(f"Collection stats: {stats}")

    return f"{collection_name}:{total_inserted}"


if __name__ == "__main__":
    from kfp import compiler

    compiler.Compiler().compile(
        ingest_to_milvus,
        package_path=__file__.replace(".py", "_component.yaml"),
    )
