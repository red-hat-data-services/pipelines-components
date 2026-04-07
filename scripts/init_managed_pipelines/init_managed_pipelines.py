"""Copy managed pipeline YAMLs to the shared volume for the KFP API server."""

import json
import shutil
from pathlib import Path

APP_ROOT = Path("/app")
OUTPUT_DIR = Path("/config/managed-pipelines")
MANIFEST = "managed-pipelines.json"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

entries = json.loads((APP_ROOT / MANIFEST).read_text())

for e in entries:
    src = (APP_ROOT / e["path"].replace("pipeline.py", "pipeline.yaml")).resolve()
    if not src.is_relative_to(APP_ROOT.resolve()):
        raise ValueError(f"Manifest path escapes app root: {e['path']}")
    if not src.is_file():
        raise FileNotFoundError(f"Pipeline YAML missing for {e['name']!r}: {src}")
    shutil.copy2(src, OUTPUT_DIR / f"{e['name']}.yaml")

shutil.copy2(APP_ROOT / MANIFEST, OUTPUT_DIR / MANIFEST)

print(f"Staged {len(entries)} pipeline(s) in {OUTPUT_DIR}")
