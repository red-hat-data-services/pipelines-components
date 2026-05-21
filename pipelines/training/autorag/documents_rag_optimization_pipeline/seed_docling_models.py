#!/usr/bin/env python3
"""Populate Docling artifact dirs from RHAI OCI model images (application/x-mlmodel).

OCI sources (registry.redhat.io/rhai):
  docling-project-docling-layout-heron:3.0  -> docling-layout-heron/
  docling-project-docling-models:3.0        -> docling-models-3.0/

Files are copied under DOCLING_ARTIFACTS_PATH using HuggingFace-cache directory names
expected by docling at runtime:

  docling-project--docling-layout-heron/
  docling-project--docling-models/

See https://docling-project.github.io/docling/usage/advanced_options/
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

# OCI artifact root folder names (inside each mlmodel image).
LAYOUT_SOURCE_NAMES = (
    "docling-layout-heron",
    "docling-project--docling-layout-heron",
)
MODELS_SOURCE_NAMES = (
    "docling-models-3.0",
    "docling-models",
    "docling-project--docling-models",
)

# Directory names docling resolves under DOCLING_ARTIFACTS_PATH.
LAYOUT_DEST_NAME = "docling-project--docling-layout-heron"
MODELS_DEST_NAME = "docling-project--docling-models"


def _resolve_bundle_dir(root: Path, known_names: tuple[str, ...]) -> Path:
    """Return the directory that contains model files under *root*."""
    if not root.is_dir():
        print(f"error: model root not found: {root}", file=sys.stderr)
        sys.exit(1)

    if (root / "config.json").is_file() or (root / "model.safetensors").is_file():
        return root

    for name in known_names:
        candidate = root / name
        if candidate.is_dir():
            return candidate

    subdirs = sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("."))
    if len(subdirs) == 1:
        return subdirs[0]

    print(
        f"error: could not resolve model bundle under {root} "
        f"(expected one of {known_names!r} or a single subdirectory)",
        file=sys.stderr,
    )
    sys.exit(1)


def _copy_tree(src: Path, dest: Path) -> int:
    """Copy *src* into *dest*, replacing any existing tree. Returns file count."""
    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dest, symlinks=False)
    return sum(1 for path in dest.rglob("*") if path.is_file())


def _install_from_oci(layout_root: Path, models_root: Path, dest: Path) -> None:
    layout_src = _resolve_bundle_dir(layout_root, LAYOUT_SOURCE_NAMES)
    models_src = _resolve_bundle_dir(models_root, MODELS_SOURCE_NAMES)
    dest.mkdir(parents=True, exist_ok=True)

    layout_files = _copy_tree(layout_src, dest / LAYOUT_DEST_NAME)
    models_files = _copy_tree(models_src, dest / MODELS_DEST_NAME)
    if layout_files == 0 or models_files == 0:
        print("error: one or more model trees are empty after copy", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    """Parse CLI arguments and populate the Docling artifact tree."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dest",
        type=Path,
        required=True,
        help="Directory that will contain docling-project--* model folders (DOCLING_ARTIFACTS_PATH).",
    )
    parser.add_argument(
        "--layout-root",
        type=Path,
        required=True,
        help="Filesystem root copied from the docling-layout-heron OCI image.",
    )
    parser.add_argument(
        "--models-root",
        type=Path,
        required=True,
        help="Filesystem root copied from the docling-models OCI image.",
    )
    args = parser.parse_args()
    _install_from_oci(args.layout_root, args.models_root, args.dest)


if __name__ == "__main__":
    main()
