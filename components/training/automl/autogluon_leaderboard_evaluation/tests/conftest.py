"""Test fixtures for leaderboard_evaluation."""

from unittest import mock

import pytest


def _make_component_status_artifact(tmp_path):
    art = mock.MagicMock()
    art.path = str(tmp_path / "component_status_out")
    art.metadata = {}
    return art


@pytest.fixture(autouse=True)
def inject_component_status(monkeypatch, tmp_path):
    """Inject component_status when tests omit it."""
    from ..component import leaderboard_evaluation

    original = leaderboard_evaluation.python_func

    def wrapper(*args, **kwargs):
        if "component_status" not in kwargs:
            kwargs["component_status"] = _make_component_status_artifact(tmp_path)
        return original(*args, **kwargs)

    monkeypatch.setattr(leaderboard_evaluation, "python_func", wrapper)
