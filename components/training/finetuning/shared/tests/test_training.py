"""Unit tests for the shared training utilities."""

import logging
from dataclasses import dataclass
from datetime import datetime
from unittest import mock

import pytest

from ..training import _log_job_details, compute_nproc, safe_int, select_runtime, wait_for_training_job


@dataclass
class FakeStep:
    """Fake TrainJob step for testing."""

    name: str
    pod_name: str
    status: str = "Running"


@dataclass
class FakeTrainJob:
    """Fake TrainJob object for testing."""

    name: str
    status: str
    steps: list
    creation_timestamp: datetime = None


@pytest.fixture
def log():
    """Create a test logger."""
    return logging.getLogger("test_training")


@pytest.fixture
def mock_client():
    """Create a mock TrainerClient with a test namespace."""
    client = mock.MagicMock()
    client.backend.namespace = "test-ns"
    return client


def _make_train_job(status="Running", steps=None, name="test-job"):
    """Create a FakeTrainJob with sensible defaults.

    Args:
        status: Job status string.
        steps: List of FakeStep instances.
        name: Job name.
    """
    if steps is None:
        steps = [FakeStep(name="node-0", pod_name="test-job-node-0-abc")]
    return FakeTrainJob(
        name=name,
        status=status,
        steps=steps,
        creation_timestamp=datetime(2026, 3, 11, 10, 0, 0),
    )


class TestSharedTrainingUnitTests:
    """Unit tests for shared training utility functions."""

    def test_log_job_details_logs_job_info(self, mock_client, log):
        """Test that job name, namespace, and status are logged."""
        train_job = _make_train_job()
        with mock.patch.object(log, "info") as mock_info:
            _log_job_details(mock_client, train_job, log, has_node_0=True)
            messages = [str(c) for c in mock_info.call_args_list]
            assert any("test-job" in m and "test-ns" in m and "Running" in m for m in messages)
            assert any("2026" in m for m in messages)

    def test_log_job_details_logs_pod_names_and_kubectl(self, mock_client, log):
        """Test that pod names and kubectl commands are logged."""
        train_job = _make_train_job(
            steps=[
                FakeStep(name="node-0", pod_name="job-node-0-abc"),
                FakeStep(name="node-1", pod_name="job-node-1-def"),
            ]
        )
        with mock.patch.object(log, "info") as mock_info:
            _log_job_details(mock_client, train_job, log, has_node_0=True)
            messages = [str(c) for c in mock_info.call_args_list]
            assert any("job-node-0-abc" in m for m in messages)
            assert any("job-node-1-def" in m for m in messages)
            assert any("kubectl -n test-ns logs job-node-0-abc -f" in m for m in messages)

    def test_log_job_details_logs_no_pods_when_steps_empty(self, mock_client, log):
        """Test that a 'no pods' message is logged when steps are empty."""
        train_job = _make_train_job(steps=[])
        with mock.patch.object(log, "info") as mock_info:
            _log_job_details(mock_client, train_job, log, has_node_0=False)
            messages = [str(c) for c in mock_info.call_args_list]
            assert any("No training pods found yet" in m for m in messages)

    def test_log_job_details_shows_streaming_message_when_has_node_0(self, mock_client, log):
        """Test that streaming message appears when node-0 exists."""
        train_job = _make_train_job()
        with mock.patch.object(log, "info") as mock_info:
            _log_job_details(mock_client, train_job, log, has_node_0=True)
            messages = [str(c) for c in mock_info.call_args_list]
            assert any("Streaming logs for node-0" in m for m in messages)

    def test_log_job_details_hides_streaming_message_when_no_node_0(self, mock_client, log):
        """Test that streaming message is absent when node-0 is missing."""
        train_job = _make_train_job(steps=[FakeStep(name="worker-0", pod_name="w-pod")])
        with mock.patch.object(log, "info") as mock_info:
            _log_job_details(mock_client, train_job, log, has_node_0=False)
            messages = [str(c) for c in mock_info.call_args_list]
            assert not any("Streaming logs for node-0" in m for m in messages)

    def test_wait_happy_path_streams_logs_and_completes(self, mock_client, log):
        """Test that logs are streamed and job completes successfully."""
        running_job = _make_train_job(status="Running")
        complete_job = _make_train_job(status="Complete")
        mock_client.get_job.side_effect = [running_job, complete_job]
        mock_client.get_job_logs.return_value = iter(["epoch 1", "epoch 2", "done"])

        wait_for_training_job(mock_client, "test-job", log)

        assert mock_client.wait_for_job_status.call_count == 2
        mock_client.wait_for_job_status.assert_any_call(name="test-job", status={"Running"}, timeout=900)
        mock_client.wait_for_job_status.assert_any_call(name="test-job", status={"Complete", "Failed"}, timeout=1800)
        mock_client.get_job_logs.assert_called_once_with(name="test-job", step="node-0", follow=True)

    def test_wait_streaming_retries_then_succeeds(self, mock_client, log):
        """Test that log streaming retries on failure and succeeds."""
        running_job = _make_train_job(status="Running")
        complete_job = _make_train_job(status="Complete")
        mock_client.get_job.side_effect = [running_job, complete_job]
        mock_client.get_job_logs.side_effect = [
            RuntimeError("container creating"),
            iter(["log line"]),
        ]

        with mock.patch("components.training.finetuning.shared.training.time.sleep"):
            wait_for_training_job(mock_client, "test-job", log)

        assert mock_client.get_job_logs.call_count == 2
        assert mock_client.wait_for_job_status.call_count == 2

    def test_wait_streaming_fails_all_retries_falls_back_to_polling(self, mock_client, log):
        """Test that polling is used when all streaming retries fail."""
        running_job = _make_train_job(status="Running")
        complete_job = _make_train_job(status="Complete")
        mock_client.get_job.side_effect = [running_job, complete_job]
        mock_client.get_job_logs.side_effect = RuntimeError("container not ready")

        with mock.patch("components.training.finetuning.shared.training.time.sleep"):
            wait_for_training_job(mock_client, "test-job", log)

        assert mock_client.get_job_logs.call_count == 3
        assert mock_client.wait_for_job_status.call_count == 2
        mock_client.wait_for_job_status.assert_any_call(name="test-job", status={"Complete", "Failed"}, timeout=1800)

    def test_wait_node_0_not_found_skips_streaming(self, mock_client, log):
        """Test that streaming is skipped when node-0 step is absent."""
        job_no_node0 = _make_train_job(
            status="Running",
            steps=[FakeStep(name="worker-0", pod_name="w-pod")],
        )
        complete_job = _make_train_job(status="Complete")
        mock_client.get_job.side_effect = [job_no_node0, complete_job]

        wait_for_training_job(mock_client, "test-job", log)

        mock_client.get_job_logs.assert_not_called()
        assert mock_client.wait_for_job_status.call_count == 2

    def test_wait_get_job_fails_skips_logging_and_streaming(self, mock_client, log):
        """Test that a get_job failure skips logging and streaming."""
        complete_job = _make_train_job(status="Complete")
        mock_client.get_job.side_effect = [
            RuntimeError("API error"),
            complete_job,
        ]

        with mock.patch.object(log, "warning") as mock_warning:
            wait_for_training_job(mock_client, "test-job", log)
            warnings = [str(c) for c in mock_warning.call_args_list]
            assert any("Could not retrieve TrainJob details" in w for w in warnings)
            assert not any("node-0 step not found" in w for w in warnings)

        mock_client.get_job_logs.assert_not_called()
        assert mock_client.wait_for_job_status.call_count == 2

    @pytest.mark.parametrize(
        "status,match",
        [
            ("Failed", "Job failed"),
            ("Suspended", "Unexpected status"),
        ],
    )
    def test_wait_job_bad_status_raises_runtime_error(self, mock_client, log, status, match):
        """Test that Failed or unexpected status raises RuntimeError."""
        running_job = _make_train_job(status="Running")
        bad_job = _make_train_job(status=status)
        mock_client.get_job.side_effect = [running_job, bad_job]
        mock_client.get_job_logs.return_value = iter([])

        with pytest.raises(RuntimeError, match=match):
            wait_for_training_job(mock_client, "test-job", log)


class TestSafeInt:
    """Tests for safe_int."""

    def test_none_returns_default(self):
        """None input returns the default value."""
        assert safe_int(None, 42) == 42

    def test_int_passthrough(self):
        """Integer input passes through unchanged."""
        assert safe_int(7, 0) == 7

    def test_string_int(self):
        """String integer is parsed correctly."""
        assert safe_int("10", 0) == 10

    def test_string_with_whitespace(self):
        """Whitespace-padded string integer is parsed."""
        assert safe_int("  5  ", 0) == 5

    def test_empty_string_returns_default(self):
        """Empty string returns the default value."""
        assert safe_int("", 99) == 99

    def test_zero(self):
        """Zero is returned, not confused with falsy default."""
        assert safe_int(0, 42) == 0

    def test_float_string_raises(self):
        """Float string raises ValueError."""
        with pytest.raises(ValueError):
            safe_int("3.14", 0)


class TestSelectRuntime:
    """Tests for select_runtime."""

    def test_finds_matching_runtime(self, log):
        """Default 'training-hub' runtime is found."""
        rt = mock.MagicMock()
        rt.name = "training-hub"
        client = mock.MagicMock()
        client.list_runtimes.return_value = [rt]

        result = select_runtime(client, log)
        assert result is rt

    def test_finds_custom_named_runtime(self, log):
        """Custom-named runtime is found when specified."""
        rt1 = mock.MagicMock()
        rt1.name = "other"
        rt2 = mock.MagicMock()
        rt2.name = "my-runtime"
        client = mock.MagicMock()
        client.list_runtimes.return_value = [rt1, rt2]

        result = select_runtime(client, log, runtime_name="my-runtime")
        assert result is rt2

    def test_raises_when_not_found(self, log):
        """RuntimeError raised when no runtimes exist."""
        client = mock.MagicMock()
        client.list_runtimes.return_value = []

        with pytest.raises(RuntimeError, match="not found"):
            select_runtime(client, log)

    def test_raises_when_no_match(self, log):
        """RuntimeError raised when target runtime name has no match."""
        rt = mock.MagicMock()
        rt.name = "wrong-name"
        client = mock.MagicMock()
        client.list_runtimes.return_value = [rt]

        with pytest.raises(RuntimeError, match="training-hub.*not found"):
            select_runtime(client, log)


class TestComputeNproc:
    """Tests for compute_nproc."""

    def test_auto_uses_gpu_count(self):
        """'auto' nproc_per_node uses GPU count."""
        np, nn = compute_nproc(4, "auto", num_workers=2)
        assert np == 4
        assert nn == 2

    def test_auto_case_insensitive(self):
        """'AUTO' is treated the same as 'auto'."""
        np, _ = compute_nproc(8, "AUTO")
        assert np == 8

    def test_explicit_nproc(self):
        """Explicit nproc_per_node string is parsed."""
        np, nn = compute_nproc(4, "2", num_workers=3)
        assert np == 2
        assert nn == 3

    def test_single_node_forces_nnodes_1(self):
        """Single-node mode forces nnodes to 1."""
        np, nn = compute_nproc(4, "auto", num_workers=8, single_node=True)
        assert np == 4
        assert nn == 1

    def test_min_values_clamped_to_1(self):
        """Zero values are clamped to minimum of 1."""
        np, nn = compute_nproc(0, "0", num_workers=0)
        assert np == 1
        assert nn == 1

    def test_default_workers(self):
        """Default num_workers produces nnodes of 1."""
        _, nn = compute_nproc(1, "auto")
        assert nn == 1
