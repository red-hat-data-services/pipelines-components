"""Unit tests for time series pipeline executor resource tiers."""

from kfp_components.utils.pipeline_task_resources import (
    assert_executor_resources,
    compile_executor_resources,
)

from ..pipeline import autogluon_timeseries_training_pipeline
from .pipeline_resource_expectations import (
    AUTOML_TIMESERIES_EXECUTOR_RESOURCES,
    TRAINING_BALANCED_RESOURCES,
    TRAINING_SPEED_RESOURCES,
)


class TestAutogluonTimeseriesPipelineResourceRequirements:
    """Time series pipeline declares preset-dependent training tiers plus shared loader/leaderboard tiers."""

    def test_timeseries_pipeline_executor_resources(self):
        """All time series pipeline executors match the declared CPU/memory matrix."""
        assert_executor_resources(
            compile_executor_resources(autogluon_timeseries_training_pipeline),
            AUTOML_TIMESERIES_EXECUTOR_RESOURCES,
            pipeline_name="autogluon_timeseries_training_pipeline",
        )

    def test_default_speed_preset_uses_lower_training_tier(self):
        """Default speed preset branch requests less CPU/memory than balanced."""
        actual = compile_executor_resources(autogluon_timeseries_training_pipeline)
        speed_keys = [name for name in actual if name.endswith("-2") and "models-training" in name]
        balanced_keys = [name for name in actual if "models-training" in name and not name.endswith("-2")]
        assert len(speed_keys) == 1
        assert len(balanced_keys) == 1
        speed = actual[speed_keys[0]]
        balanced = actual[balanced_keys[0]]
        assert speed == TRAINING_SPEED_RESOURCES
        assert balanced == TRAINING_BALANCED_RESOURCES
        assert float(speed.cpu_request) < float(balanced.cpu_request)
