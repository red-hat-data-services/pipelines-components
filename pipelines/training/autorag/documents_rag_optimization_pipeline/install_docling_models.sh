#!/usr/bin/env bash
# Install Docling model trees for offline text extraction.
#
# RHAI sources (application/x-mlmodel OCI artifacts — not valid buildah FROM images):
#   registry.redhat.io/rhai/docling-project-docling-layout-heron:3.0@sha256:8f3f5df82afb...
#   registry.redhat.io/rhai/docling-project-docling-models:3.0@sha256:ce28b996df65...
#
# Resolution order:
#   1. Hermeto generic prefetch (${HERMETO_GENERIC_DEPS}): extracted trees or .tar/.tar.gz archives
#      listed in artifacts.lock.yaml (see generate_docling_rhai_lockfile.sh).
#   2. oras copy from registry (networked builds, e.g. local Containerfile without prefetch).
#
# Hermetic Konflux builds cannot use FROM on mlmodel artifacts; models must be prefetched.

set -euo pipefail

readonly LAYOUT_REF="${DOCLING_LAYOUT_REF:-registry.redhat.io/rhai/docling-project-docling-layout-heron:3.0@sha256:8f3f5df82afb2d0bcb40110d85cf1879b9d762f1600cca79c94f2a6b7173575b}"
readonly MODELS_REF="${DOCLING_MODELS_REF:-registry.redhat.io/rhai/docling-project-docling-models:3.0@sha256:ce28b996df65c0d12e0646b038ac4d3387b19feb6d8f13d74da720e3d4ecd4ce}"
readonly DEST="${DOCLING_ARTIFACTS_DEST:-/opt/app-root/docling-artifacts/models}"
readonly WORK_ROOT="${DOCLING_WORK_ROOT:-/tmp/docling-oci}"
readonly SEED_SCRIPT="${DOCLING_SEED_SCRIPT:-/tmp/seed_docling_models.py}"

find_in_hermeto() {
  local name="$1"
  local root="${HERMETO_GENERIC_DEPS:-}"
  [[ -n "${root}" && -d "${root}" ]] || return 1
  find "${root}" \( -name "${name}" -o -name "${name}*" \) \( -type f -o -type d \) 2>/dev/null | head -1
}

extract_archive_to() {
  local archive="$1"
  local out_dir="$2"
  mkdir -p "${out_dir}"
  if tar -tf "${archive}" >/dev/null 2>&1; then
    case "${archive}" in
      *.tar.gz | *.tgz) tar -xzf "${archive}" -C "${out_dir}" ;;
      *) tar -xf "${archive}" -C "${out_dir}" ;;
    esac
    return 0
  fi
  echo "install_docling_models.sh: error: ${archive} is not a tar archive (Hermeto layer blob?)" >&2
  return 1
}

materialize_hermeto_root() {
  local logical_name="$1"
  local out_dir="$2"
  local found
  found="$(find_in_hermeto "${logical_name}" || true)"
  if [[ -z "${found}" ]]; then
    return 1
  fi
  rm -rf "${out_dir}"
  if [[ -d "${found}" ]]; then
    mkdir -p "$(dirname "${out_dir}")"
    cp -a "${found}" "${out_dir}"
    return 0
  fi
  extract_archive_to "${found}" "${out_dir}"
}

install_from_hermeto() {
  local layout_dir="${WORK_ROOT}/layout-root"
  local models_dir="${WORK_ROOT}/models-root"
  if ! materialize_hermeto_root "docling-layout-heron-3.0" "${layout_dir}"; then
    return 1
  fi
  if ! materialize_hermeto_root "docling-models-3.0" "${models_dir}"; then
    return 1
  fi
  echo "Using Docling models from Hermeto generic prefetch"
  python3 "${SEED_SCRIPT}" \
    --dest "${DEST}" \
    --layout-root "${layout_dir}" \
    --models-root "${models_dir}"
}

install_from_hermeto_file_tree() {
  local root="${HERMETO_GENERIC_DEPS:-}"
  [[ -n "${root}" && -d "${root}" ]] || return 1
  if ! find "${root}" -mindepth 1 -maxdepth 2 -type d -name 'docling-project--*' | grep -q .; then
    return 1
  fi
  echo "Using Docling models from Hermeto file tree (docling-project--*)"
  python3 "${SEED_SCRIPT}" --dest "${DEST}" --hermeto-dir "${root}"
}

install_from_oras() {
  if ! command -v oras >/dev/null 2>&1; then
    echo "install_docling_models.sh: error: oras is required to pull RHAI Docling OCI artifacts" >&2
    return 1
  fi
  local layout_dir="${WORK_ROOT}/layout-root"
  local models_dir="${WORK_ROOT}/models-root"
  rm -rf "${WORK_ROOT}"
  mkdir -p "${layout_dir}" "${models_dir}"
  echo "Pulling ${LAYOUT_REF} with oras"
  oras copy "${LAYOUT_REF}" "${layout_dir}/"
  echo "Pulling ${MODELS_REF} with oras"
  oras copy "${MODELS_REF}" "${models_dir}/"
  python3 "${SEED_SCRIPT}" \
    --dest "${DEST}" \
    --layout-root "${layout_dir}" \
    --models-root "${models_dir}"
}

main() {
  mkdir -p "$(dirname "${DEST}")"
  if install_from_hermeto; then
    return 0
  fi
  if install_from_hermeto_file_tree; then
    return 0
  fi
  if [[ -n "${HERMETO_GENERIC_DEPS:-}" && -d "${HERMETO_GENERIC_DEPS}" ]]; then
    echo "install_docling_models.sh: error: HERMETO_GENERIC_DEPS is set but Docling models were not prefetched." >&2
    echo "  Run pipelines/training/autorag/documents_rag_optimization_pipeline/generate_docling_rhai_lockfile.sh" >&2
    echo "  on a cluster with registry.redhat.io access and update artifacts.lock.yaml." >&2
    exit 1
  fi
  install_from_oras
  rm -rf "${WORK_ROOT}"
}

main "$@"
