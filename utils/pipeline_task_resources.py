"""Helpers and expected tiers for Kubernetes CPU/memory on compiled pipeline executors."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from kfp import compiler

_RESOURCE_KEYS = (
    "resourceCpuRequest",
    "resourceMemoryRequest",
    "resourceCpuLimit",
    "resourceMemoryLimit",
)


@dataclass(frozen=True)
class ExecutorResources:
    """CPU and memory requests/limits as emitted in compiled pipeline YAML."""

    cpu_request: str
    memory_request: str
    cpu_limit: str
    memory_limit: str

    @classmethod
    def from_mapping(cls, resources: Mapping[str, Any]) -> ExecutorResources:
        """Build from a container ``resources`` mapping in compiled YAML."""
        missing = [key for key in _RESOURCE_KEYS if key not in resources]
        if missing:
            msg = f"Resource mapping missing keys: {missing}"
            raise ValueError(msg)
        return cls(
            cpu_request=str(resources["resourceCpuRequest"]),
            memory_request=str(resources["resourceMemoryRequest"]),
            cpu_limit=str(resources["resourceCpuLimit"]),
            memory_limit=str(resources["resourceMemoryLimit"]),
        )


def _executor_resources_from_document(doc: dict[str, Any]) -> dict[str, ExecutorResources]:
    """Collect executor resources from one YAML document."""
    deployment_specs: list[dict[str, Any]] = []
    root_spec = doc.get("deploymentSpec")
    if isinstance(root_spec, dict):
        deployment_specs.append(root_spec)
    platform_spec = doc.get("platforms", {}).get("kubernetes", {}).get("deploymentSpec")
    if isinstance(platform_spec, dict):
        deployment_specs.append(platform_spec)

    collected: dict[str, ExecutorResources] = {}
    for deployment_spec in deployment_specs:
        executors = deployment_spec.get("executors")
        if not isinstance(executors, dict):
            continue
        for executor_name, executor in executors.items():
            if not isinstance(executor, dict):
                continue
            container = executor.get("container")
            if not isinstance(container, dict):
                continue
            raw = container.get("resources")
            if not isinstance(raw, dict) or not all(key in raw for key in _RESOURCE_KEYS):
                continue
            collected[str(executor_name)] = ExecutorResources.from_mapping(raw)
    return collected


def compile_executor_resources(pipeline_func: Any) -> dict[str, ExecutorResources]:
    """Compile *pipeline_func* and return ``exec-<task>`` -> resource mapping."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        compiler.Compiler().compile(pipeline_func=pipeline_func, package_path=tmp_path)
        import yaml

        docs = [doc for doc in yaml.safe_load_all(Path(tmp_path).read_text()) if isinstance(doc, dict)]
        merged: dict[str, ExecutorResources] = {}
        for doc in docs:
            merged.update(_executor_resources_from_document(doc))
        if not merged:
            msg = "Compiled pipeline YAML contains no executor resource definitions."
            raise ValueError(msg)
        return merged
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def normalize_executor_name(executor_name: str) -> str:
    """Strip ``exec-`` prefix from a deploymentSpec executor key."""
    if executor_name.startswith("exec-"):
        return executor_name[len("exec-") :]
    return executor_name


def assert_executor_resources(
    actual: Mapping[str, ExecutorResources],
    expected: Mapping[str, ExecutorResources],
    *,
    pipeline_name: str,
    allow_extra: bool = False,
) -> None:
    """Fail when compiled executor resources differ from *expected*.

    Compares normalized executor names (``exec-`` prefix removed). When pipeline
    resource tiers change, update the expected mapping in the unit test that
    calls this helper.

    Args:
        actual: Executor name -> resources from ``compile_executor_resources``.
        expected: Normalized task name -> expected resources.
        pipeline_name: Label included in assertion errors.
        allow_extra: When True, *actual* may contain executors not listed in *expected*
            (useful for partial assertions such as preset training tiers only).
    """
    actual_by_task = {normalize_executor_name(name): resources for name, resources in actual.items()}
    expected_by_task = dict(expected)

    missing = sorted(set(expected_by_task) - set(actual_by_task))
    extra = sorted(set(actual_by_task) - set(expected_by_task))
    if missing:
        msg = [f"{pipeline_name}: executor resource set mismatch.", f"  missing executors: {missing}"]
        raise AssertionError("\n".join(msg))
    if extra and not allow_extra:
        msg = [f"{pipeline_name}: executor resource set mismatch.", f"  unexpected executors: {extra}"]
        raise AssertionError("\n".join(msg))

    mismatches: list[str] = []
    for task_name, want in expected_by_task.items():
        got = actual_by_task[task_name]
        if got != want:
            mismatches.append(f"  {task_name}: expected {want}, got {got}")
    if mismatches:
        header = f"{pipeline_name}: executor resources changed (update tests if intentional):"
        raise AssertionError("\n".join([header, *mismatches]))
