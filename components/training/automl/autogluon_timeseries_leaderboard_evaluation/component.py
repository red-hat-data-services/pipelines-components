from pathlib import Path
from typing import List, NamedTuple

from kfp import dsl

_SHARED_DIR = Path(__file__).parent.parent / "shared"


@dsl.component(
    base_image="registry.redhat.io/rhoai/odh-pipeline-runtime-datascience-cpu-py312-rhel9@sha256:f9844dc150592a9f196283b3645dda92bd80dfdb3d467fa8725b10267ea5bdbc",  # noqa: E501
    embedded_artifact_path=str(_SHARED_DIR),
)
def timeseries_leaderboard_evaluation(
    models: List[dsl.Model],
    eval_metric: str,
    html_artifact: dsl.Output[dsl.HTML],
    embedded_artifact: dsl.EmbeddedInput[dsl.Artifact],
) -> NamedTuple("outputs", best_model=str):
    """Evaluate refitted AutoGluon timeseries models and generate a leaderboard.

    This component aggregates metrics from a list of model artifacts produced by
    ``autogluon_timeseries_models_full_refit`` (collected via ``dsl.Collected``) and
    generates an HTML-formatted leaderboard ranking the models by their performance metric.

    Each artifact in ``models`` must have been produced by
    ``autogluon_timeseries_models_full_refit``, which writes the following layout under
    the artifact path::

        {artifact.path}/{model_name_full}/metrics/metrics.json
        {artifact.path}/{model_name_full}/predictor/
        {artifact.path}/{model_name_full}/notebooks/

    Note: KFP does not propagate artifact metadata through executor inputs for collected
    artifact lists, so metrics are read from ``metrics/metrics.json`` on the filesystem
    rather than from ``artifact.metadata``.

    Args:
        models: List of model artifacts from ``autogluon_timeseries_models_full_refit``
            collected via ``dsl.Collected``. Each artifact provides metrics and location
            metadata for one refitted model.
        eval_metric: Metric name for ranking (e.g. ``"MASE"``, ``"WAPE"``);
            leaderboard is sorted descending (AutoGluon uses higher-is-better convention
            so metrics like MASE are negated - higher value means better model).
        html_artifact: Output artifact for the HTML-formatted leaderboard.
        embedded_artifact: Embedded shared files injected by the KFP runtime;
            provides ``leaderboard_html_template.html``.

    Returns:
        NamedTuple with ``best_model`` (str): display name of the top-ranked model.

    Raises:
        TypeError: If ``eval_metric`` is empty or not a string.
        ValueError: If ``models`` list is empty, or if no artifact produced a ``metrics.json``.

    Note:
        Artifacts without a ``metrics/metrics.json`` file are skipped with a warning
        (e.g. when the corresponding refit task failed).

    Example:
        from kfp import dsl
        from components.training.automl.autogluon_timeseries_leaderboard_evaluation import (
            timeseries_leaderboard_evaluation
        )

        @dsl.pipeline(name="ts-evaluation-pipeline")
        def evaluation_pipeline():
            with dsl.ParallelFor(items=selection_task.outputs["top_models"]) as model_name:
                refit_task = autogluon_timeseries_models_full_refit(model_name=model_name, ...)
            leaderboard = timeseries_leaderboard_evaluation(
                models=dsl.Collected(refit_task.outputs["model_artifact"]),
                eval_metric="MASE",
            )
    """  # noqa: E501
    import json
    import logging
    from pathlib import Path

    import pandas as pd
    from leaderboard_utils import _build_leaderboard_html, _build_leaderboard_table, _round_metrics

    logger = logging.getLogger(__name__)

    # Input validation
    if not isinstance(eval_metric, str) or not eval_metric.strip():
        raise TypeError("eval_metric must be a non-empty string.")
    if not models:
        raise ValueError("models list must not be empty.")

    results = []
    for artifact in models:
        artifact_path = Path(artifact.path)
        base_uri = artifact.uri.rstrip("/")

        # KFP does not propagate artifact metadata through executor inputs for collected
        # artifact lists — artifact.metadata is always empty here. Read from the files
        # written by autogluon_timeseries_models_full_refit instead.
        #
        # Layout written by the refit component:
        #   {artifact.path}/{model_name_full}/metrics/metrics.json
        #   {artifact.path}/{model_name_full}/predictor/
        #   {artifact.path}/{model_name_full}/notebooks/
        metrics_file = next(artifact_path.glob("*/metrics/metrics.json"), None)
        if metrics_file is None:
            logger.warning(
                "Skipping artifact at '%s': no metrics/metrics.json found. "
                "The refit task may have failed for this model.",
                artifact.uri,
            )
            continue
        model_name = metrics_file.relative_to(artifact_path).parts[0]

        with open(metrics_file, "r", encoding="utf-8") as f:
            metrics_dict = json.load(f)

        predictor_uri = f"{base_uri}/{model_name}/predictor"
        notebook_uri = f"{base_uri}/{model_name}/notebooks/automl_predictor_notebook.ipynb"

        results.append(
            {
                "model": model_name,
                **_round_metrics(metrics_dict),
                "notebook": notebook_uri,
                "predictor": predictor_uri,
            }
        )

    if not results:
        raise ValueError(
            "No valid model artifacts found. All refit tasks may have failed "
            "or produced no metrics/metrics.json output."
        )

    # Notice: AutoGluon follows the "higher is better" strategy for all metrics.
    # Metrics like MASE and WAPE have their signs flipped so that higher value = better model.
    leaderboard_df = pd.DataFrame(results).sort_values(by=eval_metric, ascending=False)
    n = len(leaderboard_df)
    leaderboard_df.index = pd.RangeIndex(start=1, stop=n + 1, name="rank")

    html_table = _build_leaderboard_table(leaderboard_df)

    best_model_name = leaderboard_df.iloc[0]["model"]
    template_path = Path(embedded_artifact.path) / "leaderboard_html_template.html"
    html_content = _build_leaderboard_html(
        template_path=template_path,
        table_html=html_table,
        eval_metric=eval_metric,
        best_model_name=best_model_name,
        num_models=len(leaderboard_df),
    )
    with open(html_artifact.path, "w", encoding="utf-8") as f:
        f.write(html_content)

    html_artifact.metadata["data"] = leaderboard_df.to_dict()
    html_artifact.metadata["display_name"] = "automl_timeseries_leaderboard"
    return NamedTuple("outputs", best_model=str)(best_model=best_model_name)


if __name__ == "__main__":
    from kfp.compiler import Compiler

    Compiler().compile(
        timeseries_leaderboard_evaluation,
        package_path=__file__.replace(".py", "_component.yaml"),
    )
