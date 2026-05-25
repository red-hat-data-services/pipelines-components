#!/usr/bin/env bash
# Emit artifacts.lock.yaml entries for RHAI Docling OCI artifacts (Hermeto generic prefetch).
#
# Prerequisites: oras CLI, jq, registry.redhat.io login on the host.
#
# Usage:
#   ./generate_docling_rhai_lockfile.sh
#
# Copy the printed YAML into artifacts.lock.yaml (keep the sqlite-autoconf entry).

set -euo pipefail

emit_artifact_layers() {
  local ref="$1"
  local repository="$2"
  local filename_prefix="$3"

  echo "# ${ref}" >&2
  local manifest
  manifest="$(oras manifest fetch "${ref}")"
  local count=0
  while IFS= read -r digest; do
    [[ -n "${digest}" ]] || continue
    count=$((count + 1))
    local algo="${digest%%:*}"
    local hash="${digest#*:}"
    local suffix=""
    if [[ "${count}" -gt 1 ]]; then
      suffix="-layer${count}"
    fi
    cat <<EOF
  - download_url: "https://registry.redhat.io/v2/${repository}/blobs/${algo}:${hash}"
    checksum: "${algo}:${hash}"
    filename: "${filename_prefix}${suffix}"
EOF
  done < <(echo "${manifest}" | jq -r '.layers[].digest')
}

if ! command -v oras >/dev/null 2>&1 || ! command -v jq >/dev/null 2>&1; then
  echo "error: oras and jq are required" >&2
  exit 1
fi

cat <<'EOF'
# Paste under artifacts: in artifacts.lock.yaml
EOF

emit_artifact_layers \
  "registry.redhat.io/rhai/docling-project-docling-layout-heron:3.0@sha256:8f3f5df82afb2d0bcb40110d85cf1879b9d762f1600cca79c94f2a6b7173575b" \
  "rhai/docling-project-docling-layout-heron" \
  "docling-layout-heron-3.0"

emit_artifact_layers \
  "registry.redhat.io/rhai/docling-project-docling-models:3.0@sha256:ce28b996df65c0d12e0646b038ac4d3387b19feb6d8f13d74da720e3d4ecd4ce" \
  "rhai/docling-project-docling-models" \
  "docling-models-3.0"
