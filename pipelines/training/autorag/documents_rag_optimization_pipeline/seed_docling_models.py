#!/usr/bin/env python3
"""Populate Docling artifact dirs for offline use (matches docling 2.73.x layout models).

Modes (mutually exclusive):
  --download          Fetch from Hugging Face (needs network).
  --hermeto-dir       Copy from Hermeto generic output (deps/generic/...); paths must match
                      artifacts.lock.yaml ``filename`` entries.
  --oci-layout-dir    Copy from Red Hat OCI artifacts (flat tree at artifact root), together with
  --oci-models-dir    ``--oci-models-dir`` (e.g. registry.stage.redhat.io/rhai/docling-project-*:3.0).

See https://docling-project.github.io/docling/usage/advanced_options/ and
https://github.com/hermetoproject/hermeto/blob/main/docs/generic.md
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

LAYOUT_REPO = "docling-project/docling-layout-heron"
LAYOUT_REV = "main"
MODELS_REPO = "docling-project/docling-models"
MODELS_REV = "v2.3.0"

# Docling offline cache layout (HF hub uses org--repo as directory name).
LAYOUT_DEST_NAME = "docling-project--docling-layout-heron"
MODELS_DEST_NAME = "docling-project--docling-models"

# Not required at runtime; skip when copying OCI exports.
_OCI_SKIP_BASENAMES = frozenset({".DS_Store", "README.md"})


def _download(dest: Path) -> None:
    from huggingface_hub import snapshot_download

    dest.mkdir(parents=True, exist_ok=True)
    layout_name = LAYOUT_REPO.replace("/", "--")
    models_name = MODELS_REPO.replace("/", "--")
    snapshot_download(
        LAYOUT_REPO,
        revision=LAYOUT_REV,
        local_dir=str(dest / layout_name),
        local_dir_use_symlinks=False,
    )
    snapshot_download(
        MODELS_REPO,
        revision=MODELS_REV,
        local_dir=str(dest / models_name),
        local_dir_use_symlinks=False,
    )


def _from_hermeto(source: Path, dest: Path) -> None:
    """Hermeto stores files under deps/generic/ using lockfile ``filename`` (may include subdirs).

    Only paths whose first component starts with ``docling-project--`` are copied so other generic
    artifacts (e.g. SQLite source tarballs) can share the same Hermeto lockfile without landing
    under ``DOCLING_ARTIFACTS_PATH``.
    """
    if not source.is_dir():
        print(f"error: Hermeto directory not found: {source}", file=sys.stderr)
        sys.exit(1)
    dest.mkdir(parents=True, exist_ok=True)
    copied = 0
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(source)
        if rel.parts and not rel.parts[0].startswith("docling-project--"):
            continue
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        copied += 1
    if copied == 0:
        print(f"error: no files under {source}", file=sys.stderr)
        sys.exit(1)


def _resolve_oci_artifact_root(mount_dir: Path) -> Path:
    """Return directory that contains config.json (artifact root or single child dir)."""
    if (mount_dir / "config.json").is_file():
        return mount_dir
    for child in sorted(mount_dir.iterdir()):
        if child.is_dir() and (child / "config.json").is_file():
            return child
    print(f"error: no config.json under OCI mount {mount_dir}", file=sys.stderr)
    sys.exit(1)


def _copy_oci_tree(source_root: Path, dest_dir: Path) -> int:
    """Copy model files from a flat OCI artifact tree into a docling-project--* directory."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for path in sorted(source_root.rglob("*")):
        if not path.is_file():
            continue
        if path.name in _OCI_SKIP_BASENAMES:
            continue
        rel = path.relative_to(source_root)
        target = dest_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        copied += 1
    return copied


def _from_oci(layout_mount: Path, models_mount: Path, dest: Path) -> None:
    """Map Red Hat OCI artifact mounts to Docling's docling-project--* directory names."""
    dest.mkdir(parents=True, exist_ok=True)
    layout_root = _resolve_oci_artifact_root(layout_mount)
    models_root = _resolve_oci_artifact_root(models_mount)
    layout_n = _copy_oci_tree(layout_root, dest / LAYOUT_DEST_NAME)
    models_n = _copy_oci_tree(models_root, dest / MODELS_DEST_NAME)
    if layout_n == 0 or models_n == 0:
        print(
            f"error: OCI copy incomplete (layout={layout_n}, models={models_n})",
            file=sys.stderr,
        )
        sys.exit(1)


def main() -> None:
    """Parse CLI arguments and populate the Docling artifact tree."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dest",
        type=Path,
        required=True,
        help="Directory that will contain docling-project--* model folders (same as DOCLING_ARTIFACTS_PATH).",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--download",
        action="store_true",
        help="Download pinned repos from Hugging Face Hub.",
    )
    group.add_argument(
        "--hermeto-dir",
        type=Path,
        metavar="DIR",
        help="Hermeto generic deps directory (e.g. .../deps/generic).",
    )
    parser.add_argument(
        "--oci-layout-dir",
        type=Path,
        metavar="DIR",
        help="Mount point for docling-project-docling-layout-heron OCI artifact (with --oci-models-dir).",
    )
    parser.add_argument(
        "--oci-models-dir",
        type=Path,
        metavar="DIR",
        help="Mount point for docling-project-docling-models OCI artifact (with --oci-layout-dir).",
    )
    args = parser.parse_args()
    oci = args.oci_layout_dir is not None or args.oci_models_dir is not None
    if oci and (args.oci_layout_dir is None or args.oci_models_dir is None):
        parser.error("--oci-layout-dir and --oci-models-dir must be used together")
    modes = sum([args.download, args.hermeto_dir is not None, oci])
    if modes != 1:
        parser.error("choose exactly one of --download, --hermeto-dir, or --oci-layout-dir/--oci-models-dir")
    if args.download:
        _download(args.dest)
    elif args.hermeto_dir is not None:
        _from_hermeto(args.hermeto_dir, args.dest)
    else:
        _from_oci(args.oci_layout_dir, args.oci_models_dir, args.dest)


if __name__ == "__main__":
    main()
