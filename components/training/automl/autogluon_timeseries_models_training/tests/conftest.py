"""Test fixtures for autogluon_timeseries_models_training."""

from unittest import mock

import pytest


@pytest.fixture
def component_status_artifact(tmp_path):
    """Explicit component_status output artifact for python_func calls."""
    art = mock.MagicMock()
    art.path = str(tmp_path / "component_status_out")
    art.metadata = {}
    return art
