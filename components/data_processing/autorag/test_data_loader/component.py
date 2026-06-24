from pathlib import Path

from kfp import dsl
from kfp_components.utils.consts import AUTORAG_IMAGE  # pyright: ignore[reportMissingImports]

_AUTORAG_SHARED = Path(__file__).parents[3] / "training" / "autorag" / "shared"


@dsl.component(
    base_image=AUTORAG_IMAGE,  # noqa: E501
    embedded_artifact_path=str(_AUTORAG_SHARED / "component_status.py"),
)
def test_data_loader(
    test_data_bucket_name: str,
    test_data_path: str,
    benchmark_sample_size: int = 25,
    test_data: dsl.Output[dsl.Artifact] = None,
    component_status: dsl.Output[dsl.Artifact] = None,
    embedded_artifact: dsl.EmbeddedInput[dsl.Dataset] = None,
):
    """Download test data JSON from S3 and sample it for benchmarking.

    Thin wrapper that delegates to ``ai4rag.components.data.test_data_loader.load_test_data``.

    Args:
        test_data_bucket_name: S3 (or compatible) bucket that contains the test
            data file.
        test_data_path: S3 object key to the JSON test data file.
        benchmark_sample_size: Maximum number of records to keep from the test
            data. When the dataset exceeds this limit, a reproducible random
            sample is drawn (seed 42). Set to 0 to disable sampling and keep
            all records.
        test_data: Output artifact that receives the (possibly sampled) file.
        component_status: Output artifact containing stage-level progress tracking.
        embedded_artifact: Embedded ``autorag.shared`` helpers injected by KFP at runtime.

    Environment variables (required when run with pipeline secret injection):
        AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_S3_ENDPOINT.
        AWS_DEFAULT_REGION is optional.
    """
    import importlib.util
    import json
    import logging
    from pathlib import Path

    from ai4rag.components.data.test_data_loader import load_test_data
    from ai4rag.components.utils.s3 import create_s3_client

    logging.basicConfig(level=logging.INFO)

    if component_status is None:
        from kfp_components.components.training.autorag.shared.component_status import (  # pyright: ignore[reportMissingImports]
            null_component_status_tracker,
        )

        status = null_component_status_tracker()
    else:
        _embedded_path = Path(embedded_artifact.path)
        _module_path = _embedded_path if _embedded_path.is_file() else _embedded_path / "component_status.py"
        _spec = importlib.util.spec_from_file_location("_autorag_component_status", _module_path)
        if _spec is None or _spec.loader is None:
            raise ValueError(f"Cannot load embedded module from {_module_path}")
        _status_module = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_status_module)
        status = _status_module.bootstrap_status_tracker(embedded_artifact, component_status, "test_data_loader")
    with status:
        if component_status is not None:
            status.set_metadata(display_name="Test Data Loader Status")
            component_status.metadata["display_name"] = "Test Data Loader Status"
        with status.stage("load_benchmark"):
            s3_client = create_s3_client()

            result = load_test_data(
                bucket_name=test_data_bucket_name,
                key=test_data_path,
                benchmark_sample_size=benchmark_sample_size,
                s3_client=s3_client,
            )

            with open(test_data.path, "w", encoding="utf-8") as f:
                json.dump(result.data, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    from kfp.compiler import Compiler

    Compiler().compile(
        test_data_loader,
        package_path=__file__.replace(".py", "_component.yaml"),
    )
