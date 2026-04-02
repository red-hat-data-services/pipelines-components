import sys
from pathlib import Path

# Add the shared automl directory to sys.path so that the component body's bare import
# `from leaderboard_utils import ...` resolves during testing, replicating the KFP
# runtime behaviour where embedded_artifact_path is added to sys.path inside the container.
# parents[0]=tests, [1]=component dir, [2]=automl — shared lives at automl/shared
_shared_dir = str(Path(__file__).resolve().parents[2] / "shared")
if _shared_dir not in sys.path:
    sys.path.insert(0, _shared_dir)
