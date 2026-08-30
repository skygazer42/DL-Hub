from dataclasses import dataclass
import os
from pathlib import Path, PureWindowsPath


_WINDOWS_INVALID_CHARS = frozenset('<>:"|?*')


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


def _normalize_component(value: str, *, field: str) -> str:
    """Return one portable path component or reject unsafe input."""

    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string, got {type(value).__name__}")

    normalized = value.strip().replace(" ", "_")
    if not normalized:
        raise ValueError(f"{field} must not be empty")

    # Keep every caller-supplied value to exactly one directory level. Checking
    # both separators is intentional: a backslash is harmless on POSIX but is a
    # separator on Windows, so accepting it would make the API platform-dependent.
    if "/" in normalized or "\\" in normalized or normalized in {".", ".."}:
        raise ValueError(f"{field} must be a single path component, got {value!r}")

    if any(ord(char) < 32 or ord(char) == 127 for char in normalized):
        raise ValueError(f"{field} must not contain control characters")

    if any(char in _WINDOWS_INVALID_CHARS for char in normalized) or normalized.endswith("."):
        raise ValueError(f"{field} is not a portable path component: {value!r}")

    windows_path = PureWindowsPath(normalized)
    if windows_path.drive or windows_path.is_reserved():
        raise ValueError(f"{field} is not a portable path component: {value!r}")

    return normalized


def build_run_paths(track: str, lesson: str, run_name: str) -> RunPaths:
    """Build output paths without allowing components to escape the output root."""

    safe_track = _normalize_component(track, field="track")
    safe_lesson = _normalize_component(lesson, field="lesson")
    safe_name = _normalize_component(run_name, field="run_name")

    run_dir = get_outputs_root() / safe_track / safe_lesson / safe_name
    checkpoints_dir = run_dir / "checkpoints"
    logs_dir = run_dir / "logs"
    return RunPaths(run_dir=run_dir, checkpoints_dir=checkpoints_dir, logs_dir=logs_dir)
