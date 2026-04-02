from dataclasses import dataclass
import os
from pathlib import Path


@dataclass(frozen=True)
class RunPaths:
    run_dir: Path
    checkpoints_dir: Path
    logs_dir: Path


def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def get_outputs_root() -> Path:
    """Return the base directory for run outputs.

    By default this is `<repo_root>/outputs/`. Tests can override it via the
    `DLHUB_OUTPUTS_DIR` environment variable.
    """

    raw = os.environ.get("DLHUB_OUTPUTS_DIR")
    if raw:
        p = Path(raw).expanduser()
        if not p.is_absolute():
            p = get_repo_root() / p
        return p
    return get_repo_root() / "outputs"


def build_run_paths(track: str, lesson: str, run_name: str) -> RunPaths:
    safe_track = track.strip().replace(" ", "_")
    safe_lesson = lesson.strip().replace(" ", "_")
    safe_name = run_name.strip().replace(" ", "_")

    run_dir = get_outputs_root() / safe_track / safe_lesson / safe_name
    checkpoints_dir = run_dir / "checkpoints"
    logs_dir = run_dir / "logs"
    return RunPaths(run_dir=run_dir, checkpoints_dir=checkpoints_dir, logs_dir=logs_dir)
