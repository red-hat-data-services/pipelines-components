#!/usr/bin/env python3
"""Populate Docling artifact dirs for offline use (matches docling 2.73.x layout models).

Two modes:
  --download     Fetch from Hugging Face (needs network; used by default image builds).
  --hermeto-dir  Copy from Hermeto generic output (deps/generic/...); paths must match
                 artifacts.lock.yaml ``filename`` entries (networkless Konflux builds).

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


def main() -> None:
    """Parse CLI arguments and populate the Docling artifact tree."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dest",
        type=Path,
        required=True,
        help="Directory that will contain docling-project--* model folders (same as DOCLING_ARTIFACTS_PATH).",
    )
    group = parser.add_mutually_exclusive_group(required=True)
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
    args = parser.parse_args()
    if args.download:
        _download(args.dest)
    else:
        _from_hermeto(args.hermeto_dir, args.dest)


if __name__ == "__main__":
    main()
