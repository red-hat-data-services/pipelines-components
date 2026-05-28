"""Pipeline manifest publisher for AutoML pipelines.

Publishes the pipeline structure (components, stages, steps) as an artifact at the start
of a pipeline run so dashboards/UIs know the expected execution plan before components run.
"""

from kfp import dsl
from kfp_components.utils.consts import AUTOML_IMAGE  # pyright: ignore[reportMissingImports]


@dsl.component(
    base_image=AUTOML_IMAGE,
)
def publish_pipeline_manifest(
    pipeline_id: str,
    run_id: str,
    pipeline_manifest: dsl.Output[dsl.Artifact],
) -> None:
    """Publish the pipeline structure manifest for dashboard consumption.

    Reads the static JSON manifest from the package (run_status_templates/pipelines/)
    and publishes it as a KFP artifact. This enables dashboards to show expected
    components, stages, and steps before pipeline execution begins.

    The manifest defines the complete pipeline structure:
    - Component list and execution order
    - Stages within each component
    - Steps within each stage (optional)
    - Descriptions for UI display

    Args:
        pipeline_id: Pipeline identifier matching manifest filename
                    (e.g., "autogluon-tabular-training-pipeline").
        run_id: KFP run ID for tracking (from dsl.PIPELINE_JOB_ID_PLACEHOLDER).
        pipeline_manifest: Output artifact containing the full pipeline structure.

    Raises:
        FileNotFoundError: If manifest file for pipeline_id doesn't exist.
        ValueError: If pipeline_id or run_id is empty.

    Example:
        manifest_task = publish_pipeline_manifest(
            pipeline_id="autogluon-tabular-training-pipeline",
            run_id=dsl.PIPELINE_JOB_ID_PLACEHOLDER,
        )
    """
    import json
    from datetime import UTC, datetime
    from pathlib import Path

    # Validate inputs
    if not isinstance(pipeline_id, str) or not pipeline_id.strip():
        raise ValueError("pipeline_id must be a non-empty string")
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("run_id must be a non-empty string")

    # Load manifest from package
    # Path is relative to this file: ../../shared/run_status_templates/pipelines/
    manifest_dir = Path(__file__).resolve().parent.parent / "shared" / "run_status_templates" / "pipelines"
    manifest_path = manifest_dir / f"{pipeline_id}.json"

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Pipeline manifest not found: {manifest_path}. Expected manifest file for pipeline_id='{pipeline_id}'"
        )

    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    # Add runtime metadata
    manifest["kfp_run_id"] = run_id
    manifest["published_at"] = datetime.now(UTC).isoformat().replace("+00:00", "Z")

    # Publish as artifact
    output_path = Path(pipeline_manifest.path)
    output_path.mkdir(parents=True, exist_ok=True)

    output_file = output_path / "pipeline_manifest.json"
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # Set artifact metadata for KFP UI
    pipeline_manifest.metadata["display_name"] = "Pipeline Manifest"
    pipeline_manifest.metadata["pipeline_id"] = pipeline_id
    pipeline_manifest.metadata["component_count"] = len(manifest.get("components", []))

    # Log summary
    component_count = len(manifest.get("components", []))
    stage_count = sum(len(c.get("stages", [])) for c in manifest.get("components", []))
    print(f"Published manifest for pipeline_id='{pipeline_id}':")
    print(f"  - Components: {component_count}")
    print(f"  - Total stages: {stage_count}")
    print(f"  - Published to: {output_file}")
