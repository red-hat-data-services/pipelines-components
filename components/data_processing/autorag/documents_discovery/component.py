from pathlib import Path

from kfp import dsl
from kfp_components.utils.consts import AUTORAG_IMAGE  # pyright: ignore[reportMissingImports]

_AUTORAG_SHARED = Path(__file__).parents[3] / "training" / "autorag" / "shared"


@dsl.component(
    base_image=AUTORAG_IMAGE,  # noqa: E501
    embedded_artifact_path=str(_AUTORAG_SHARED / "component_status.py"),
    install_kfp_package=False,
)
def documents_discovery(
    input_data_bucket_name: str,
    input_data_path: str = "",
    test_data: dsl.Input[dsl.Artifact] = None,
    sampling_enabled: bool = True,
    sampling_max_size: float = 1,
    discovered_documents: dsl.Output[dsl.Artifact] = None,
    component_status: dsl.Output[dsl.Artifact] = None,
    embedded_artifact: dsl.EmbeddedInput[dsl.Dataset] = None,
):
    """Documents discovery component.

    Lists available documents from S3, performs sampling if applied and writes a JSON manifest
    (documents_descriptor.json) with metadata. Does not download document contents.

    Args:
        input_data_bucket_name: S3 (or compatible) bucket containing input data.
        input_data_path: Path to folder with input documents within the bucket.
        test_data: Optional input artifact containing test data for sampling.
        sampling_enabled: Whether to enable sampling or not.
        sampling_max_size: Maximum size of sampled documents (in gigabytes).
        discovered_documents: Output artifact containing the documents descriptor JSON file.
        component_status: Output artifact containing stage-level progress tracking.
        embedded_artifact: Embedded ``autorag.shared`` helpers injected by KFP at runtime.

    Environment variables (required when run with pipeline secret injection):
        AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_S3_ENDPOINT.
        AWS_DEFAULT_REGION is optional.
    """
    import json
    import logging
    import os
    import sys
    from math import inf
    from pathlib import Path

    import boto3

    logger = logging.getLogger("Document Loader component logger")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(handler)

    DOCUMENTS_DESCRIPTOR_FILENAME = "documents_descriptor.json"
    SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".pptx", ".md", ".html", ".txt"}
    MAX_SIZE_BYTES = float(inf)

    if sampling_enabled:
        MAX_SIZE_BYTES = float(sampling_max_size) * 1024**3

    def get_test_data_docs_names() -> list[str]:
        if test_data is None:
            return []
        with open(test_data.path, "r") as f:
            benchmark = json.load(f)

        docs_names = []
        for question in benchmark:
            docs_names.extend(question["correct_answer_document_ids"])

        return docs_names

    import importlib.util

    from botocore.exceptions import SSLError

    _embedded_path = Path(embedded_artifact.path)
    _module_path = _embedded_path if _embedded_path.is_file() else _embedded_path / "component_status.py"
    _spec = importlib.util.spec_from_file_location("_autorag_component_status", _module_path)
    if _spec is None or _spec.loader is None:
        raise ValueError(f"Cannot load embedded module from {_module_path}")
    _status_module = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_status_module)
    status = _status_module.bootstrap_status_tracker(embedded_artifact, component_status, "documents_discovery")
    with status:
        with status.stage("validate_inputs"):
            s3_creds = {k: os.environ.get(k) for k in ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_S3_ENDPOINT"]}
            for k, v in s3_creds.items():
                if v is None:
                    raise ValueError(
                        "%s environment variable not set. Check if kubernetes secret was configured properly" % k
                    )
            s3_creds["AWS_DEFAULT_REGION"] = os.environ.get("AWS_DEFAULT_REGION")

            def _make_s3_client(verify=True):
                return boto3.client(
                    "s3",
                    endpoint_url=s3_creds["AWS_S3_ENDPOINT"],
                    region_name=s3_creds["AWS_DEFAULT_REGION"],
                    aws_access_key_id=s3_creds["AWS_ACCESS_KEY_ID"],
                    aws_secret_access_key=s3_creds["AWS_SECRET_ACCESS_KEY"],
                    verify=verify,
                )

        with status.stage("list_and_sample"):
            # Use paginator to handle buckets with >1,000 objects
            def _list_all_objects(s3_client):
                """List all objects under prefix using pagination."""
                paginator = s3_client.get_paginator("list_objects_v2")
                contents = []
                for page in paginator.paginate(
                    Bucket=input_data_bucket_name,
                    Prefix=input_data_path,
                ):
                    contents.extend(page.get("Contents", []))
                return contents

            try:
                s3_client = _make_s3_client()
                contents = _list_all_objects(s3_client)
            except SSLError:
                logger.warning(
                    "SSL error when listing objects in s3://%s/%s, retrying with verify=False",
                    input_data_bucket_name,
                    input_data_path,
                )
                s3_client = _make_s3_client(verify=False)
                contents = _list_all_objects(s3_client)

            logger.info(
                "S3_DISCOVERY bucket=%s prefix=%s total_objects=%d",
                input_data_bucket_name,
                input_data_path,
                len(contents),
            )

            supported_files = [c for c in contents if c["Key"].endswith(tuple(SUPPORTED_EXTENSIONS))]
            if not supported_files:
                raise Exception("No supported documents found.")

            test_data_docs_names = get_test_data_docs_names()
            if test_data_docs_names:
                test_keys_set = {c["Key"] for c in supported_files if Path(c["Key"]).name in test_data_docs_names}
                supported_files.sort(key=lambda c: c["Key"] not in test_keys_set)

            total_size = 0
            selected = []
            for file in supported_files:
                if total_size + file["Size"] > MAX_SIZE_BYTES:
                    continue
                selected.append(file)
                total_size += file["Size"]

            documents = []
            for file_info in selected:
                key = file_info["Key"]
                size_bytes = file_info["Size"]
                documents.append(
                    {
                        "key": key,
                        "size_bytes": size_bytes,
                    }
                )
            if not documents:
                raise ValueError(
                    "No documents to process. Check that the bucket/prefix is correct and contains supported files."
                )

            descriptor = {
                "bucket": input_data_bucket_name,
                "prefix": input_data_path,
                "documents": documents,
                "total_size_bytes": total_size,
                "count": len(documents),
            }

            logger.info(
                "DISCOVERY_COMPLETE bucket=%s prefix=%s document_count=%d total_size_bytes=%d sampling=%s",
                input_data_bucket_name,
                input_data_path,
                len(documents),
                total_size,
                f"enabled_max={sampling_max_size}GB" if sampling_enabled else "disabled",
            )

        with status.stage("write_descriptor"):
            os.makedirs(discovered_documents.path, exist_ok=True)
            descriptor_path = os.path.join(discovered_documents.path, DOCUMENTS_DESCRIPTOR_FILENAME)
            with open(descriptor_path, "w") as f:
                json.dump(descriptor, f, indent=2)

            logger.info("Documents descriptor written to %s", descriptor_path)


if __name__ == "__main__":
    from kfp.compiler import Compiler

    Compiler().compile(
        documents_discovery,
        package_path=__file__.replace(".py", "_component.yaml"),
    )
