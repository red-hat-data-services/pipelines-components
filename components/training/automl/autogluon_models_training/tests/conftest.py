"""Test fixtures for autogluon_models_training."""

from unittest import mock

import pytest


@pytest.fixture(autouse=True)
def inject_component_status(monkeypatch, tmp_path):
    """Inject component_status when tests omit it."""
    from ..component import autogluon_models_training

    original = autogluon_models_training.python_func

    def wrapper(*args, **kwargs):
        if "component_status" not in kwargs:
            art = mock.MagicMock()
            art.path = str(tmp_path / "component_status_out")
            art.metadata = {}
            kwargs["component_status"] = art
        return original(*args, **kwargs)

    monkeypatch.setattr(autogluon_models_training, "python_func", wrapper)
