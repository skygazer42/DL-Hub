from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunPaths:
    run_dir: Path
    checkpoints_dir: Path
    logs_dir: Path


def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def build_run_paths(track: str, lesson: str, run_name: str) -> RunPaths:
    safe_track = track.strip().replace(" ", "_")
    safe_lesson = lesson.strip().replace(" ", "_")
    safe_name = run_name.strip().replace(" ", "_")

    run_dir = get_repo_root() / "outputs" / safe_track / safe_lesson / safe_name
    checkpoints_dir = run_dir / "checkpoints"
    logs_dir = run_dir / "logs"
    return RunPaths(run_dir=run_dir, checkpoints_dir=checkpoints_dir, logs_dir=logs_dir)
