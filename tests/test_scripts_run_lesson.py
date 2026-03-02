from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_run_lesson_lists_tracks() -> None:
    proc = subprocess.run(
        [sys.executable, "scripts/run_lesson.py", "--list"],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "Tracks:" in proc.stdout
    assert "vision" in proc.stdout


def test_run_lesson_lists_lessons_for_track() -> None:
    proc = subprocess.run(
        [sys.executable, "scripts/run_lesson.py", "vision", "--list"],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "Lessons (vision):" in proc.stdout
    assert "lesson_01_mnist_lenet" in proc.stdout


def test_run_lesson_dry_run_resolves_train_module() -> None:
    proc = subprocess.run(
        [sys.executable, "scripts/run_lesson.py", "vision", "lesson_01_mnist_lenet", "--dry-run"],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.vision.lesson_01_mnist_lenet.train" in proc.stdout

