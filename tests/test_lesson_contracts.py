import json
import subprocess
import sys
from pathlib import Path

from scripts.lesson_contracts import (
    OFFLINE_BUILT_IN,
    OFFLINE_EXPLICIT_FAKE,
    SOURCE_INLINE,
    SOURCE_LOCAL,
    SOURCE_NOT_APPLICABLE,
    discover_curated_smoke_lessons,
    discover_lesson_contracts,
    get_lesson_contract,
    validate_lesson_contracts,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_lesson_contracts_are_valid() -> None:
    errors = validate_lesson_contracts(_repo_root())
    assert errors == []


def test_required_readmes_and_alternate_structure_sources_are_explicit() -> None:
    contracts = discover_lesson_contracts(_repo_root())
    assert len(contracts) == 339
    assert all(contract.has_readme for contract in contracts)

    alternate_models = {
        contract.key: contract.model_source
        for contract in contracts
        if contract.model_source != SOURCE_LOCAL
    }
    assert alternate_models == {
        ("foundations", "lesson_01_tensors"): SOURCE_NOT_APPLICABLE,
        (
            "vision",
            "lesson_15_neural_style_transfer_gatys",
        ): "dlhub.vision.style_transfer_zoo",
        (
            "vision",
            "lesson_16_style_transfer_translation_cyclegan",
        ): "dlhub.vision.style_transfer_zoo",
    }

    alternate_data = {
        contract.key: contract.data_source
        for contract in contracts
        if contract.data_source != SOURCE_LOCAL
    }
    assert alternate_data == {
        ("foundations", "lesson_01_tensors"): SOURCE_INLINE,
        ("gnn", "lesson_05_label_propagation_cora"): "tracks.gnn.datasets.cora",
        ("gnn", "lesson_06_graphsage_cora"): "tracks.gnn.datasets.cora",
    }

    for contract in contracts:
        if "." in contract.model_source:
            assert contract.model_source in contract.imported_modules
        if "." in contract.data_source:
            assert contract.data_source in contract.imported_modules


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
    assert "required README.md: 339/339 present (0 missing)" in proc.stdout
    assert "model sources: 336 local, 2 shared, 1 not-applicable, 0 missing" in proc.stdout
    assert "data sources: 336 local, 2 shared, 1 inline, 0 missing" in proc.stdout
    assert "lesson contracts: OK" in proc.stdout


def test_contract_json_check_is_one_machine_readable_document() -> None:
    proc = subprocess.run(
        [sys.executable, "scripts/lesson_contracts.py", "--json", "--check"],
        cwd=_repo_root(),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["ok"] is True
    assert payload["errors"] == []
    assert len(payload["contracts"]) == len(discover_lesson_contracts(_repo_root()))


def test_contract_records_importability_and_executable_guard(tmp_path: Path) -> None:
    track_dir = tmp_path / "tracks" / "demo"
    lesson_dir = track_dir / "lesson_01_example"
    lesson_dir.mkdir(parents=True)
    (track_dir / "__init__.py").write_text("", encoding="utf-8")
    (lesson_dir / "__init__.py").write_text("", encoding="utf-8")
    (lesson_dir / "run.py").write_text(
        "def run():\n    return 0\n\nif '__main__' == __name__:\n    raise SystemExit(run())\n",
        encoding="utf-8",
    )

    contract = get_lesson_contract("demo", "lesson_01_example", tmp_path)
    assert contract is not None
    assert contract.has_track_init is True
    assert contract.has_main_guard is True


def test_contract_validation_requires_readme_model_and_data_sources(tmp_path: Path) -> None:
    track_dir = tmp_path / "tracks" / "demo"
    lesson_dir = track_dir / "lesson_01_example"
    lesson_dir.mkdir(parents=True)
    (track_dir / "__init__.py").write_text("", encoding="utf-8")
    (lesson_dir / "__init__.py").write_text("", encoding="utf-8")
    (lesson_dir / "run.py").write_text(
        "def main():\n    return 0\n\nif __name__ == '__main__':\n    raise SystemExit(main())\n",
        encoding="utf-8",
    )

    errors = validate_lesson_contracts(tmp_path)
    assert "demo/lesson_01_example: missing required README.md" in errors
    assert any("missing model.py without a declared" in error for error in errors)
    assert any("missing data.py without a declared" in error for error in errors)


def test_contract_lookup_rejects_non_module_paths(tmp_path: Path) -> None:
    assert get_lesson_contract("..", "lesson_01_example", tmp_path) is None
    assert get_lesson_contract("demo", "../lesson_01_example", tmp_path) is None


def test_contract_validation_reports_unparseable_smoke_module(tmp_path: Path) -> None:
    smoke_dir = tmp_path / "scripts" / "smoke_checks"
    smoke_dir.mkdir(parents=True)
    (smoke_dir / "broken.py").write_text("def broken(:\n", encoding="utf-8")

    errors = validate_lesson_contracts(tmp_path)
    assert any("cannot parse smoke module" in error for error in errors)
    assert "lesson tracks directory is missing: tracks" in errors


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
