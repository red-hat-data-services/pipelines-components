"""Test fixtures for tabular_data_loader."""

from unittest import mock

import pytest


def _make_component_status_artifact(tmp_path):
    art = mock.MagicMock()
    art.path = str(tmp_path / "component_status")
    art.metadata = {}
    return art


@pytest.fixture(autouse=True)
def inject_component_status(monkeypatch, tmp_path):
    """Inject component_status when tests omit it."""
    from ..component import automl_data_loader

    original = automl_data_loader.python_func

    def wrapper(*args, **kwargs):
        kwargs.setdefault("component_status", _make_component_status_artifact(tmp_path))
        return original(*args, **kwargs)

    monkeypatch.setattr(automl_data_loader, "python_func", wrapper)
