"""Example pipelines demonstrating usage of leaderboard_evaluation."""

from kfp import dsl
from kfp_components.components.training.automl.autogluon_leaderboard_evaluation import (
    leaderboard_evaluation,
)


@dsl.pipeline(
    name="autogluon-leaderboard-evaluation-example",
    pipeline_config=dsl.PipelineConfig(
        workspace=dsl.WorkspaceConfig(
            size="1Gi",
            kubernetes=dsl.KubernetesWorkspaceConfig(
                pvcSpecPatch={
                    "accessModes": ["ReadWriteOnce"],
                }
            ),
        ),
    ),
)
def example_pipeline(
    eval_metric: str = "root_mean_squared_error",
):
    """Example pipeline using leaderboard_evaluation.

    Args:
        eval_metric: Evaluation metric name.
    """
    models_artifact = dsl.importer(
        artifact_uri="gs://placeholder/models_artifact",
        artifact_class=dsl.Model,
    )
    leaderboard_evaluation(
        models_artifact=models_artifact.output,
        eval_metric=eval_metric,
    )
    # component_status output is created by KFP when composed in a pipeline.
