"""High-level integration tests for AutoGluon tabular training pipeline on RHOAI.

These tests require a Red Hat OpenShift AI (RHOAI) cluster with Data Science Pipelines
enabled, and environment variables set for cluster URL, credentials, and S3 storage.
See the conftest.py in this directory for required env vars. When not set, tests
are skipped unless STRICT=true (then get_rhoai_config raises with missing env
names). You can set vars via a .env file (see .env.template).

Scenarios are parametrized via test_configs: each config specifies dataset
location, target column, problem type, AutoML/pipeline settings, and optional
tags. Filter by tags with RHOAI_TEST_CONFIG_TAGS (e.g. smoke, regression).
"""

import re
import secrets
import sys
import time
from datetime import datetime, timedelta, timezone

import kfp_server_api
import pytest
from integration_config import RHOAI_INTEGRATION_CONFIG, RHOAI_INTEGRATION_SKIP_REASON
from test_configs import get_test_configs_for_run, resolve_config_to_pipeline_arguments

# Configs to run this session (all, or filtered by RHOAI_TEST_CONFIG_TAGS).
CONFIGS_FOR_RUN = get_test_configs_for_run()

# Pipeline display name in KFP (from pipeline decorator)
PIPELINE_DISPLAY_NAME = "autogluon-tabular-training-pipeline"

# Seconds between status lines while waiting on the remote pipeline run.
PIPELINE_STATUS_POLL_INTERVAL_SEC = 20

# Hide KFP/Argo scaffolding (drivers, root DAG, executor, loops); keep DSL component names
# like ``automl-data-loader``, ``models-selection`` (see `pipeline.yaml` task ``name`` fields).
_UUID_ONLY_DISPLAY_NAME = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


def _should_omit_task_from_progress(display_name: str) -> bool:
    """Return True for infra tasks (not user components in compiled ``pipeline.yaml``)."""
    if not display_name:
        return True
    name = display_name.strip()
    if _UUID_ONLY_DISPLAY_NAME.match(name):
        return True
    key = name.lower()
    if key.endswith("-driver"):
        return True
    if key in ("root", "executor"):
        return True
    if key.startswith("for-loop"):
        return True
    if key.startswith("iteration-item"):
        return True
    if key.startswith("iteration-iterations"):
        return True
    # Compiled root DAG pod: pipeline display name + random suffix (see pipeline decorator / YAML)
    if re.match(rf"^{re.escape(PIPELINE_DISPLAY_NAME)}-[a-z0-9]+$", key):
        return True
    return False


def _task_display_label(task) -> str:
    """Prefer API display_name; fall back to task_id (used for filtering and printing)."""
    return (getattr(task, "display_name", None) or getattr(task, "task_id", None) or "").strip()


def _emit_pipeline_progress_lines(lines: list[str], capfd: pytest.CaptureFixture[str]) -> None:
    """Print progress to the real terminal while pytest capture is enabled.

    ``TerminalReporter.write_line`` is buffered until the test ends; use the ``capfd``
    fixture and ``capfd.disabled()`` so stderr reaches the console during long waits.
    """
    with capfd.disabled():
        for line in lines:
            print(line, file=sys.stderr, flush=True)


def _runtime_state_str(state):
    """Normalize KFP/runtime state for display (API may return str or enum-like values)."""
    if state is None:
        return "unknown"
    if isinstance(state, str):
        return state
    return str(state)


def _format_task_detail_lines(detail):
    """Build one line per *component* task from ``run_details.task_details`` (skip infra)."""
    run_details = getattr(detail, "run_details", None)
    if not run_details:
        return ["  (no run_details yet; tasks appear as the pipeline schedules work)"]
    tasks = getattr(run_details, "task_details", None) or []
    if not tasks:
        return ["  (no task_details yet)"]
    component_tasks = [t for t in tasks if not _should_omit_task_from_progress(_task_display_label(t))]
    if not component_tasks:
        return [
            "  (no component tasks in run_details yet; only drivers/root/executor/loops)",
        ]
    lines = []
    for task in sorted(
        component_tasks,
        key=lambda t: (_task_display_label(t).lower(), getattr(t, "task_id", "") or ""),
    ):
        name = _task_display_label(task) or "?"
        tid = getattr(task, "task_id", None) or ""
        st = _runtime_state_str(getattr(task, "state", None))
        pod = getattr(task, "pod_name", None)
        chunks = [f"  - {name}", f"state={st}"]
        if tid:
            chunks.append(f"task_id={tid}")
        if pod:
            chunks.append(f"pod={pod}")
        err = getattr(task, "error", None)
        if err is not None:
            msg = getattr(err, "message", None) or str(err)
            if msg:
                short = msg.replace("\n", " ").strip()
                if len(short) > 120:
                    short = short[:117] + "..."
                chunks.append(f"error={short}")
        lines.append(" ".join(chunks))
    return lines


def _make_automl_run_name():
    """Return a run name: automl-test-<6 hex chars>-<YYYYMMDD-HHMMSS>."""
    hex_part = secrets.token_hex(3)  # 3 bytes -> 6 hex chars
    time_part = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"automl-test-{hex_part}-{time_part}"


def _wait_for_run_with_progress(client, run_id, timeout, capfd):
    """Poll the run until a terminal state or timeout.

    Every PIPELINE_STATUS_POLL_INTERVAL_SEC, emits the run state and one line per task
    from ``run_details.task_details`` (name, state, task_id, pod, optional error) using
    ``capfd.disabled()`` so output is visible without ``pytest -s``. Mirrors kfp wait.
    """
    start = datetime.now()
    if isinstance(timeout, timedelta):
        timeout = timeout.total_seconds()
    finish_states = ("succeeded", "failed", "skipped", "error")
    is_valid_token = False
    while True:
        try:
            detail = client.get_run(run_id)
            is_valid_token = True
        except kfp_server_api.ApiException as api_ex:
            if is_valid_token and api_ex.status == 401:
                client._refresh_api_client_token()
                continue
            raise

        state = detail.state
        elapsed = (datetime.now() - start).total_seconds()
        state_str = _runtime_state_str(state)
        progress_lines = [
            f"[pipeline run {run_id}] state={state_str} elapsed_s={elapsed:.0f}",
            *_format_task_detail_lines(detail),
        ]
        _emit_pipeline_progress_lines(progress_lines, capfd)
        if elapsed > timeout:
            raise TimeoutError("Run timeout")
        if state is not None and str(state).lower() in finish_states:
            return detail
        time.sleep(PIPELINE_STATUS_POLL_INTERVAL_SEC)


def _run_pipeline_and_wait(client, compiled_path, arguments, timeout, capfd):
    """Submit pipeline run and wait for completion; return run_id and run detail."""
    run_name = _make_automl_run_name()
    run = client.create_run_from_pipeline_package(
        compiled_path,
        arguments=arguments,
        run_name=run_name,
    )
    run_id = run.run_id
    _emit_pipeline_progress_lines([""], capfd)
    detail = _wait_for_run_with_progress(client, run_id, timeout, capfd)
    _emit_pipeline_progress_lines([""], capfd)
    return run_id, detail


def _run_succeeded(detail):
    """Return True if the run finished successfully."""
    run = getattr(detail, "run", detail)
    state = getattr(run, "state", None)
    if state is None and hasattr(run, "status"):
        state = getattr(run.status, "state", None)
    if isinstance(state, str):
        return state.upper() == "SUCCEEDED"
    return False


def _find_artifacts_in_s3(s3_client, bucket, prefix):
    """List object keys under prefix; return lists of keys ending in .pkl, .ipynb, and more.

    Also returns keys containing 'leaderboard' or 'html_artifact'.
    """
    pkl_keys = []
    ipynb_keys = []
    leaderboard_keys = []
    try:
        paginator = s3_client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents") or []:
                key = obj["Key"]
                if key.endswith(".pkl"):
                    pkl_keys.append(key)
                elif key.endswith(".ipynb"):
                    ipynb_keys.append(key)
                elif "leaderboard" in key.lower() or "html_artifact" in key.lower():
                    leaderboard_keys.append(key)
    except Exception:
        pass
    return pkl_keys, ipynb_keys, leaderboard_keys


@pytest.mark.integration
@pytest.mark.skipif(
    RHOAI_INTEGRATION_CONFIG is None,
    reason=RHOAI_INTEGRATION_SKIP_REASON
    or "RHOAI integration env not set (see .env.template)",
)
@pytest.mark.parametrize("test_config", CONFIGS_FOR_RUN, ids=[c.id for c in CONFIGS_FOR_RUN])
class TestAutogluonPipelineIntegration:
    """Integration tests running the pipeline on RHOAI and validating outcomes."""

    def test_autogluon_pipeline_with_config(
        self,
        capfd,
        test_config,
        rhoai_integration_config,
        rhoai_project,
        uploaded_datasets,
        kfp_client,
        compiled_pipeline_path,
        pipeline_run_timeout,
        s3_client,
    ):
        """Run pipeline for one test config; assert success and presence of artifacts."""
        if not uploaded_datasets or not kfp_client:
            pytest.skip("Integration prerequisites not available")
        if test_config.problem_type == "timeseries":
            pytest.skip("Timeseries not yet supported by pipeline or test data")
        config = rhoai_integration_config
        arguments = resolve_config_to_pipeline_arguments(
            test_config, uploaded_datasets, config["s3_secret_name"]
        )
        if not arguments:
            pytest.skip(f"Dataset not available for path: {test_config.dataset_path}")

        run_id, detail = _run_pipeline_and_wait(
            kfp_client,
            compiled_pipeline_path,
            arguments,
            pipeline_run_timeout,
            capfd,
        )
        assert _run_succeeded(detail), (
            f"Pipeline run {run_id} did not succeed; state={getattr(detail, 'run', detail)}"
        )

        if s3_client and config.get("s3_bucket_artifacts"):
            bucket = config["s3_bucket_artifacts"]
            prefix = f"{PIPELINE_DISPLAY_NAME}/{run_id}"
            pkl_keys, ipynb_keys, leaderboard_keys = _find_artifacts_in_s3(
                s3_client, bucket, prefix
            )
            assert len(pkl_keys) >= 1, (
                f"Expected at least one .pkl model artifact under {prefix}; found {pkl_keys}"
            )
            assert len(ipynb_keys) >= 1, (
                f"Expected at least one .ipynb notebook under {prefix}; found {ipynb_keys}"
            )
            assert len(leaderboard_keys) >= 1, (
                f"Expected leaderboard/html artifact under {prefix}; found {leaderboard_keys}"
            )
