import subprocess
import sys
from pathlib import Path

from scripts.lesson_contracts import (
    OFFLINE_BUILT_IN,
    OFFLINE_EXPLICIT_FAKE,
    discover_curated_smoke_lessons,
    discover_lesson_contracts,
    validate_lesson_contracts,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_lesson_contracts_are_valid() -> None:
    errors = validate_lesson_contracts(_repo_root())
    assert errors == []


def test_offline_interfaces_and_curated_smoke_are_explicit() -> None:
    contracts = discover_lesson_contracts(_repo_root())
    assert contracts
    assert {contract.offline_mode for contract in contracts} == {
        OFFLINE_BUILT_IN,
        OFFLINE_EXPLICIT_FAKE,
    }

    all_tracks = {contract.track for contract in contracts}
    covered_tracks = {track for track, _ in discover_curated_smoke_lessons(_repo_root())}
    assert covered_tracks == all_tracks


def test_contract_cli_check() -> None:
    proc = subprocess.run(
        [sys.executable, "scripts/lesson_contracts.py", "--check"],
        cwd=_repo_root(),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "lesson contracts: OK" in proc.stdout


def test_smoke_list_is_read_only_and_reports_coverage() -> None:
    proc = subprocess.run(
        [sys.executable, "scripts/smoke_check.py", "--list"],
        cwd=_repo_root(),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Curated smoke coverage:" in proc.stdout
    assert "multimodal/lesson_01_clip_compact_retrieval" in proc.stdout
