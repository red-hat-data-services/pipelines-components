"""RHOAI integration test config: load .env and build config from environment.

Used by conftest.py (fixtures) and test_pipeline_integration.py (skipif) so
skip logic and fixtures share one source of truth. Import this module instead
of conftest to avoid resolving the repo-root conftest when running tests.

Authentication: set RHOAI_TOKEN (e.g. a service account token for Jenkins/CI;
long-lived, no oc or kubeconfig required).

With STRICT=true, incomplete required env raises RuntimeError at import time
(listing missing variable names); otherwise missing env yields None and tests skip
with RHOAI_INTEGRATION_SKIP_REASON listing missing names.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from repo root (cwd) and from this directory.
# Load tests/.env last with override=True so it takes precedence (e.g. RHOAI_CREATE_DSPA).
load_dotenv()
load_dotenv(Path(__file__).resolve().parent / ".env", override=True)

RHOAI_URL_ENV = "RHOAI_URL"
RHOAI_KFP_URL_ENV = "RHOAI_KFP_URL"
RHOAI_TOKEN_ENV = "RHOAI_TOKEN"
RHOAI_PROJECT_ENV = "RHOAI_PROJECT_NAME"
S3_ENDPOINT_ENV = "AWS_S3_ENDPOINT"
S3_ACCESS_KEY_ENV = "AWS_ACCESS_KEY_ID"
S3_SECRET_KEY_ENV = "AWS_SECRET_ACCESS_KEY"
S3_REGION_ENV = "AWS_DEFAULT_REGION"
S3_INTERNAL_ENDPOINT_ENV = "AWS_S3_INTERNAL_ENDPOINT"
S3_BUCKET_DATA_ENV = "RHOAI_TEST_DATA_BUCKET"
S3_BUCKET_ARTIFACTS_ENV = "RHOAI_TEST_ARTIFACTS_BUCKET"
S3_SECRET_NAME_ENV = "RHOAI_TEST_S3_SECRET_NAME"

# Optional: create DataSciencePipelinesApplication CR in the test namespace (default: false)
# Set to "true" or "1" to create a DSPA instance via Kubernetes CustomObjectsApi.
RHOAI_CREATE_DSPA_ENV = "RHOAI_CREATE_DSPA"
# Optional: override DSPA CRD identity (defaults for Open Data Hub / RHOAI)
DSPA_API_GROUP_ENV = "RHOAI_DSPA_API_GROUP"
DSPA_API_VERSION_ENV = "RHOAI_DSPA_API_VERSION"
DSPA_PLURAL_ENV = "RHOAI_DSPA_PLURAL"
# Optional: route name prefix to identify the pipeline API route (default: ds-pipeline)
DSPA_ROUTE_NAME_PREFIX_ENV = "RHOAI_DSPA_ROUTE_NAME_PREFIX"
# Optional: seconds to wait for the route to appear after creating DSPA (default: 300)
DSPA_ROUTE_WAIT_TIMEOUT_ENV = "RHOAI_DSPA_ROUTE_WAIT_TIMEOUT"
# Optional: seconds to wait for DSPA status to become Ready (default: 600)
DSPA_READY_WAIT_TIMEOUT_ENV = "RHOAI_DSPA_READY_WAIT_TIMEOUT"
# Optional: extra seconds to wait after DSPA is Ready before using the API (default: 30)
DSPA_READY_BUFFER_SECONDS_ENV = "RHOAI_DSPA_READY_BUFFER_SECONDS"
# When set to a truthy value, missing required RHOAI/S3 env raises from get_rhoai_config
# instead of returning None (see is_strict_integration).
STRICT_INTEGRATION_ENV = "STRICT"


def is_strict_integration() -> bool:
    """Return True if STRICT env requests hard failure when integration is not configured."""
    raw = (os.environ.get(STRICT_INTEGRATION_ENV) or "").strip().lower().strip("'\"")
    return raw in ("1", "true", "yes")


def _env_nonempty(val: str | None) -> bool:
    """True if val is set and not whitespace-only."""
    return bool((val or "").strip())


def _rhoai_incomplete_strict_message(missing: list[str]) -> str:
    joined = ", ".join(missing)
    return (
        f"RHOAI integration environment is incomplete (STRICT=true). "
        f"Missing or empty: {joined}"
    )


def _rhoai_incomplete_skip_message(missing: list[str]) -> str:
    joined = ", ".join(missing)
    return (
        f"RHOAI integration environment is incomplete. Missing or empty: {joined}. "
        f"(see .env.template)"
    )


def _parse_rhoai_config_from_env() -> tuple[dict | None, list[str]]:
    """Return (config dict or None, list of missing required env var names).

    If STRICT is set and any required variable is missing, raises RuntimeError.
    """
    url = os.environ.get(RHOAI_URL_ENV)
    kfp_url = os.environ.get(RHOAI_KFP_URL_ENV)
    token = os.environ.get(RHOAI_TOKEN_ENV)
    project = os.environ.get(RHOAI_PROJECT_ENV)
    endpoint = os.environ.get(S3_ENDPOINT_ENV)
    access = os.environ.get(S3_ACCESS_KEY_ENV)
    secret = os.environ.get(S3_SECRET_KEY_ENV)
    region = os.environ.get(S3_REGION_ENV)
    internal_endpoint = os.environ.get(S3_INTERNAL_ENDPOINT_ENV)
    bucket_data = os.environ.get(S3_BUCKET_DATA_ENV)
    bucket_artifacts = os.environ.get(S3_BUCKET_ARTIFACTS_ENV)
    secret_name = os.environ.get(S3_SECRET_NAME_ENV)

    required_pairs = (
        (RHOAI_URL_ENV, url),
        (RHOAI_TOKEN_ENV, token),
        (RHOAI_PROJECT_ENV, project),
        (S3_ENDPOINT_ENV, endpoint),
        (S3_ACCESS_KEY_ENV, access),
        (S3_SECRET_KEY_ENV, secret),
        (S3_REGION_ENV, region),
        (S3_BUCKET_DATA_ENV, bucket_data),
        (S3_BUCKET_ARTIFACTS_ENV, bucket_artifacts),
        (S3_SECRET_NAME_ENV, secret_name),
    )
    missing = [name for name, val in required_pairs if not _env_nonempty(val)]
    if missing:
        if is_strict_integration():
            raise RuntimeError(_rhoai_incomplete_strict_message(missing))
        return None, missing
    return {
        "rhoai_url": url.rstrip("/"),
        "rhoai_kfp_url": kfp_url.rstrip("/") if kfp_url is not None else None,
        "rhoai_token": token.strip(),
        "rhoai_project": project.strip(),
        "s3_endpoint": endpoint,
        "s3_internal_endpoint": internal_endpoint,
        "s3_access_key": access,
        "s3_secret_key": secret,
        "s3_region": region,
        "s3_bucket_data": bucket_data,
        "s3_bucket_artifacts": bucket_artifacts,
        "s3_secret_name": secret_name,
    }, []


def get_rhoai_config():
    """Build integration config from environment; None if not configured.

    All variables used here must be set except RHOAI_KFP_URL and
    AWS_S3_INTERNAL_ENDPOINT (optional). Includes RHOAI_PROJECT_NAME,
    AWS_DEFAULT_REGION, RHOAI_TEST_ARTIFACTS_BUCKET, and RHOAI_TEST_S3_SECRET_NAME.

    If STRICT is truthy and any required variable is missing or empty, raises
    RuntimeError listing the missing environment variable names.

    When not strict and env is incomplete, returns None (see RHOAI_INTEGRATION_SKIP_REASON
    set at module load).
    """
    config, _missing = _parse_rhoai_config_from_env()
    return config


def get_dspa_config():
    """
    Return DataSciencePipelinesApplication creation config from env, or None if disabled.

    When RHOAI_CREATE_DSPA is set to a truthy value, returns a dict with:
    - create: True
    - api_group, api_version, plural: CRD identity (with defaults for Open Data Hub / RHOAI)
    """
    raw = (os.environ.get(RHOAI_CREATE_DSPA_ENV) or "").strip().lower().strip("'\"")
    if raw not in ("1", "true", "yes"):
        return None
    return {
        "create": True,
        "api_group": os.environ.get(DSPA_API_GROUP_ENV) or "datasciencepipelinesapplications.opendatahub.io",
        "api_version": os.environ.get(DSPA_API_VERSION_ENV) or "v1",
        "plural": os.environ.get(DSPA_PLURAL_ENV) or "datasciencepipelinesapplications",
        "route_name_prefix": os.environ.get(DSPA_ROUTE_NAME_PREFIX_ENV) or "ds-pipeline",
        "route_wait_timeout": int(os.environ.get(DSPA_ROUTE_WAIT_TIMEOUT_ENV) or "300"),
        "ready_wait_timeout": int(os.environ.get(DSPA_READY_WAIT_TIMEOUT_ENV) or "600"),
        "ready_buffer_seconds": int(os.environ.get(DSPA_READY_BUFFER_SECONDS_ENV) or "30"),
    }



# Single source of truth for skipif: tests run only when config is not None.
RHOAI_INTEGRATION_CONFIG, _RHOAI_INTEGRATION_MISSING_ENV = _parse_rhoai_config_from_env()
RHOAI_INTEGRATION_SKIP_REASON = (
    _rhoai_incomplete_skip_message(_RHOAI_INTEGRATION_MISSING_ENV)
    if RHOAI_INTEGRATION_CONFIG is None
    else ""
)
