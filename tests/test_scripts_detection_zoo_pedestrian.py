import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_detection_zoo_lists_pedestrian_presets() -> None:
    proc = subprocess.run(
        [sys.executable, "scripts/detection_zoo.py", "--list", "--search", "pedestrian"],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "dldet:pedestrian_fcos" in proc.stdout


def test_detection_zoo_smoke_all_pedestrian_presets_backward() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/detection_zoo.py",
            "--smoke-all",
            "--search",
            "pedestrian",
            "--backward",
            "--batch-size",
            "1",
            "--image-size",
            "64",
            "--num-classes",
            "1",
            "--width-mult",
            "0.5",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr

