from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import pytest

import scripts.lesson_entrypoint_check as entrypoint_check


@pytest.mark.parametrize("workers", [0, -1, 1.5, math.nan, math.inf, -math.inf])
def test_entrypoint_audit_rejects_invalid_worker_counts(workers: object) -> None:
    with pytest.raises(ValueError, match="workers must be at least 1"):
        entrypoint_check.audit_lesson_entrypoints(workers=workers)  # type: ignore[arg-type]


@pytest.mark.parametrize("timeout", [0, -1, math.nan, math.inf, -math.inf])
def test_entrypoint_audit_rejects_invalid_timeouts(timeout: float) -> None:
    with pytest.raises(ValueError, match="timeout_seconds must be positive and finite"):
        entrypoint_check.audit_lesson_entrypoints(timeout_seconds=timeout)


def test_entrypoint_audit_rejects_an_empty_contract_inventory(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(entrypoint_check, "discover_lesson_contracts", lambda root: [])

    with pytest.raises(ValueError, match="no lesson contracts discovered"):
        entrypoint_check.audit_lesson_entrypoints(root=tmp_path)


def test_entrypoint_checker_supports_direct_script_help() -> None:
    root = entrypoint_check.repo_root()
    process = subprocess.run(
        [sys.executable, "scripts/lesson_entrypoint_check.py", "--help"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert process.returncode == 0, process.stdout + process.stderr
    assert "--workers" in process.stdout
    assert "--timeout" in process.stdout


def test_run_only_lesson_exposes_module_help() -> None:
    root = entrypoint_check.repo_root()
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "tracks.foundations.lesson_01_tensors.run",
            "--help",
        ],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert process.returncode == 0, process.stdout + process.stderr
    assert "usage:" in process.stdout.lower()
    assert "tensor shapes and operations" in process.stdout
