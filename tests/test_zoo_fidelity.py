import json
import subprocess
import sys
from pathlib import Path

from dlhub.zoo_fidelity import (
    AUDIT_PRESSURE_BASELINE_REGISTRATIONS,
    FidelityLevel,
    MAX_REGISTRATIONS_PER_AUDITED_ARTIFACT,
    fidelity_for_artifact,
    get_fidelity_record,
    iter_fidelity_records,
    summarize_audit_pressure,
    summarize_fidelity,
    validate_audit_pressure,
    validate_fidelity_records,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_first_fidelity_audit_is_machine_checkable() -> None:
    records = iter_fidelity_records()

    assert len(records) >= 13
    assert validate_fidelity_records(REPO_ROOT) == []
    assert all(record.level is not FidelityLevel.UNREVIEWED for record in records)
    assert all(record.missing_mechanisms for record in records)

    summary = summarize_fidelity(records)
    assert summary["audited_groups"] == len(records)
    assert summary["audited_artifacts"] >= 80
    assert summary[FidelityLevel.BASELINE_ALIAS.value] >= 1
    assert summary[FidelityLevel.COMPACT.value] >= 1


def test_registration_growth_is_bounded_by_audit_depth() -> None:
    pressure = summarize_audit_pressure(AUDIT_PRESSURE_BASELINE_REGISTRATIONS)
    audited = int(pressure["audited_artifacts"])
    allowed = int(MAX_REGISTRATIONS_PER_AUDITED_ARTIFACT * audited)

    assert validate_audit_pressure(allowed) == []
    errors = validate_audit_pressure(allowed + audited)
    assert len(errors) == 1
    assert "audit more source artifacts" in errors[0]


def test_fidelity_lookup_distinguishes_reviewed_from_unreviewed() -> None:
    assert (
        fidelity_for_artifact("dlhub/vision/detection/swin_detr.py") is FidelityLevel.BASELINE_ALIAS
    )
    assert fidelity_for_artifact("dlhub/vision/backbones/resnet.py") is FidelityLevel.UNREVIEWED

    detection = get_fidelity_record("vision.detection.detr-paper-labels")
    assert "dlhub/vision/detection/open_vocab_detr.py" in detection.artifacts
    assert any("window" in mechanism.lower() for mechanism in detection.missing_mechanisms)
    assert any("open-vocabulary" in mechanism.lower() for mechanism in detection.missing_mechanisms)


def test_fidelity_cli_check_and_json_inventory() -> None:
    check = subprocess.run(
        [sys.executable, "scripts/model_fidelity.py", "--check"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert check.returncode == 0, check.stdout + check.stderr
    assert "zoo fidelity: OK" in check.stdout
    assert "Unlisted artifacts are unreviewed" in check.stdout
    assert "audit pressure:" in check.stdout

    inventory = subprocess.run(
        [sys.executable, "scripts/model_fidelity.py", "--json"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert inventory.returncode == 0, inventory.stdout + inventory.stderr
    payload = json.loads(inventory.stdout)
    assert payload["summary"]["audited_groups"] >= 13
    assert payload["audit_pressure"] is None
    assert payload["scope"] == "audited-groups-only"
    assert any(
        record["key"] == "vision.temporal-action-localization.shared-gru"
        for record in payload["records"]
    )
