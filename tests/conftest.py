import sys
from pathlib import Path

# `tracks/`, `Llms/` and other lesson packages are intentionally not
# distributed (pyproject packages only `dlhub*`); they are meant to be
# imported from the repo root, so pytest needs the root on sys.path.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
