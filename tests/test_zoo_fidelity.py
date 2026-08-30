import importlib
import json
import subprocess
import sys
from pathlib import Path

import torch

from dlhub.vision._shared.retrieval import CompactRetrievalModel
from dlhub.zoo_fidelity import (
    AUDIT_PRESSURE_BASELINE_ARTIFACTS,
    AUDIT_PRESSURE_BASELINE_REGISTRATIONS,
    AUDIT_PRESSURE_RATCHET_ARTIFACTS,
    BASELINE_INVENTORY_PATH,
    BASELINE_WRAPPER_DEBT_BASELINE,
    FidelityLevel,
    MAX_REGISTRATIONS_PER_AUDITED_ARTIFACT,
    build_baseline_inventory,
    discover_baseline_wrappers,
    fidelity_for_artifact,
    get_fidelity_record,
    iter_fidelity_records,
    summarize_audit_pressure,
    summarize_fidelity,
    validate_audit_pressure,
    validate_baseline_inventory,
    validate_fidelity_records,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SHARED_RETRIEVAL_SOURCE = "dlhub/vision/_shared/retrieval.py"
RETRIEVAL_AUDITS: dict[str, tuple[str, tuple[str, ...]]] = {
    "vision.image-retrieval.mechanism-aware-compact": (
        "image_retrieval",
        (
            "arc",
            "clipret",
            "contrastive",
            "delg",
            "gem",
            "netvlad",
            "pairret",
            "proxy",
            "regional",
            "transformerret",
        ),
    ),
    "vision.visual-place-recognition.mechanism-aware-compact": (
        "visual_place_recognition",
        (
            "apgem_vpr",
            "cosplace",
            "delg_vpr",
            "geoclip_vpr",
            "mambavpr",
            "mixvpr",
            "pairvpr",
            "patchnetvlad",
            "regionvpr",
            "transvpr",
        ),
    ),
    "vision.fine-grained-retrieval.mechanism-aware-compact": (
        "fine_grained_retrieval",
        (
            "bilinear_fgret",
            "descriptor_fgret",
            "fgclip_retr",
            "granule_retr",
            "mamba_fgret",
            "partvlad",
            "prompt_fgret",
            "regional_fgret",
            "tokenpart_retr",
            "transformer_fgret",
        ),
    ),
}


def test_fidelity_audit_is_machine_checkable() -> None:
    records = iter_fidelity_records()

    assert len(records) >= 16
    assert validate_fidelity_records(REPO_ROOT) == []
    assert all(record.level is not FidelityLevel.UNREVIEWED for record in records)
    assert all(record.missing_mechanisms for record in records)

    summary = summarize_fidelity(records)
    assert summary["audited_groups"] == len(records)
    assert summary["audited_artifacts"] >= AUDIT_PRESSURE_RATCHET_ARTIFACTS
    assert summary[FidelityLevel.BASELINE_ALIAS.value] >= 1
    assert summary[FidelityLevel.COMPACT.value] >= 1


def test_fidelity_ledger_keys_artifacts_and_evidence_are_unique_and_grounded() -> None:
    records = iter_fidelity_records()
    keys = [record.key for record in records]
    artifacts = [artifact for record in records for artifact in record.artifacts]

    assert len(keys) == len(set(keys))
    assert len(artifacts) == len(set(artifacts))
    for record in records:
        assert record.evidence
        assert record.missing_mechanisms
        assert all((REPO_ROOT / path).is_file() for path in record.artifacts)
        assert all((REPO_ROOT / path).is_file() for path in record.evidence)


def test_retrieval_audits_cover_exact_mechanism_aware_sources() -> None:
    for key, (package, stems) in RETRIEVAL_AUDITS.items():
        record = get_fidelity_record(key)
        expected_artifacts = tuple(f"dlhub/vision/{package}/{stem}.py" for stem in stems)

        assert record.level is FidelityLevel.COMPACT
        assert record.artifacts == expected_artifacts
        assert record.evidence == (
            f"dlhub/vision/{package}/_common.py",
            SHARED_RETRIEVAL_SOURCE,
            "tests/test_dlhub_retrieval_mechanisms.py",
        )
        assert len(record.missing_mechanisms) >= 2
        assert all(
            fidelity_for_artifact(path) is FidelityLevel.COMPACT
            for path in expected_artifacts
        )


def test_retrieval_audited_builders_execute_declared_compact_mechanisms() -> None:
    torch.manual_seed(20260830)
    image = torch.randn(2, 3, 32, 32)
    gallery = torch.randn(3, 3, 32, 32)
    reference_variant_specs: tuple[dict[str, int], ...] | None = None
    mechanisms = set()

    for key, (_, stems) in RETRIEVAL_AUDITS.items():
        for artifact, stem in zip(get_fidelity_record(key).artifacts, stems, strict=True):
            module_name = Path(artifact).with_suffix("").as_posix().replace("/", ".")
            module = importlib.import_module(module_name)
            builders = [
                value
                for name, value in vars(module).items()
                if name.startswith("build_")
                and callable(value)
                and getattr(value, "__module__", None) == module_name
            ]
            assert len(builders) == 1
            variants = getattr(module, "_VARIANTS")
            tiny_variant = next(name for name in variants if name.endswith("_tiny"))
            variant_specs = tuple(variants.values())
            if reference_variant_specs is None:
                reference_variant_specs = variant_specs
            else:
                assert variant_specs == reference_variant_specs

            torch.manual_seed(17)
            model = builders[0](in_channels=3, variant=tiny_variant, width_mult=0.5).eval()
            assert type(model) is CompactRetrievalModel
            assert model.family == stem
            mechanisms.add(model.mechanism)

            with torch.no_grad():
                outputs = model(image, gallery)
            assert outputs["embedding"].shape == (2, 64)
            assert outputs["gallery_embedding"].shape == (3, 64)
            torch.testing.assert_close(
                outputs["embedding"].norm(dim=1), torch.ones(2), rtol=1e-5, atol=1e-6
            )
            assert outputs["similarity"].shape == (2, 3)
            assert torch.isfinite(outputs["similarity"]).all()

    assert len(mechanisms) == 30


def test_registration_growth_is_bounded_by_audit_depth() -> None:
    pressure = summarize_audit_pressure(AUDIT_PRESSURE_BASELINE_REGISTRATIONS)
    audited = int(pressure["audited_artifacts"])
    allowed = int(MAX_REGISTRATIONS_PER_AUDITED_ARTIFACT * audited)

    assert validate_audit_pressure(allowed) == []
    errors = validate_audit_pressure(allowed + audited)
    assert len(errors) == 1
    assert "audit more source artifacts" in errors[0]


def test_retrieval_audit_lowers_and_locks_the_pressure_ratchet() -> None:
    original_pressure = (
        AUDIT_PRESSURE_BASELINE_REGISTRATIONS / AUDIT_PRESSURE_BASELINE_ARTIFACTS
    )
    current = summarize_audit_pressure(AUDIT_PRESSURE_BASELINE_REGISTRATIONS)
    audited = int(current["audited_artifacts"])
    allowed = int(MAX_REGISTRATIONS_PER_AUDITED_ARTIFACT * audited)

    assert audited >= AUDIT_PRESSURE_RATCHET_ARTIFACTS
    assert MAX_REGISTRATIONS_PER_AUDITED_ARTIFACT == (
        AUDIT_PRESSURE_BASELINE_REGISTRATIONS / AUDIT_PRESSURE_RATCHET_ARTIFACTS
    )
    assert float(current["registrations_per_audited_artifact"]) < original_pressure
    assert float(current["registrations_per_audited_artifact"]) <= (
        MAX_REGISTRATIONS_PER_AUDITED_ARTIFACT
    )
    assert validate_audit_pressure(allowed) == []
    assert validate_audit_pressure(allowed + 1)


def test_fidelity_lookup_distinguishes_reviewed_from_unreviewed() -> None:
    assert (
        fidelity_for_artifact("dlhub/vision/detection/swin_detr.py") is FidelityLevel.BASELINE_ALIAS
    )
    assert fidelity_for_artifact("dlhub/vision/backbones/resnet.py") is FidelityLevel.UNREVIEWED

    detection = get_fidelity_record("vision.detection.detr-paper-labels")
    assert "dlhub/vision/detection/open_vocab_detr.py" in detection.artifacts
    assert any("window" in mechanism.lower() for mechanism in detection.missing_mechanisms)
    assert any("open-vocabulary" in mechanism.lower() for mechanism in detection.missing_mechanisms)


def test_baseline_wrapper_discovery_uses_direct_return_calls(tmp_path: Path) -> None:
    package = tmp_path / "dlhub" / "vision" / "example"
    package.mkdir(parents=True)
    (package / "direct.py").write_text(
        "def build():\n"
        "    return build_baseline_model(family='direct')\n",
        encoding="utf-8",
    )
    (package / "indirect.py").write_text(
        "def build():\n"
        "    model = build_baseline_model(family='indirect')\n"
        "    return model\n",
        encoding="utf-8",
    )

    wrappers = discover_baseline_wrappers(tmp_path)

    assert len(wrappers) == 1
    assert wrappers[0].artifact == "dlhub/vision/example/direct.py"
    assert wrappers[0].helper == "build_baseline_model"
    assert wrappers[0].line == 2


def test_baseline_inventory_covers_every_current_wrapper() -> None:
    wrappers = discover_baseline_wrappers(REPO_ROOT)
    inventory = build_baseline_inventory(REPO_ROOT)
    entries = inventory["wrappers"]
    summary = inventory["summary"]

    assert len(wrappers) <= BASELINE_WRAPPER_DEBT_BASELINE - 72
    assert len({wrapper.artifact for wrapper in wrappers}) == len(wrappers)
    assert summary["total_wrappers"] == len(wrappers)
    assert summary["debt_baseline"] == BASELINE_WRAPPER_DEBT_BASELINE
    assert summary["source_inferred_alias_wrappers"] > 0
    assert summary["unreviewed_wrappers"] == 0
    assert (
        summary["audited_wrappers"]
        + summary["source_inferred_alias_wrappers"]
        + summary["unreviewed_wrappers"]
        == len(wrappers)
    )
    assert len(entries) == len(wrappers)
    assert all(entry["helper"].startswith("build_baseline_") for entry in entries)
    assert all(
        entry["review_status"] in {"reviewed", "source-inferred"}
        for entry in entries
    )
    assert all(entry["level"] in {level.value for level in FidelityLevel} for entry in entries)


def test_checked_in_baseline_inventory_is_current_and_complete() -> None:
    inventory_path = REPO_ROOT / BASELINE_INVENTORY_PATH

    assert inventory_path.is_file()
    assert validate_baseline_inventory(REPO_ROOT) == []

    payload = json.loads(inventory_path.read_text(encoding="utf-8"))
    assert payload == build_baseline_inventory(REPO_ROOT)
    assert payload["summary"]["total_wrappers"] <= BASELINE_WRAPPER_DEBT_BASELINE


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
    assert "baseline wrappers:" in check.stdout

    inventory = subprocess.run(
        [sys.executable, "scripts/model_fidelity.py", "--json"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert inventory.returncode == 0, inventory.stdout + inventory.stderr
    payload = json.loads(inventory.stdout)
    assert payload["summary"]["audited_groups"] >= 16
    assert payload["summary"]["audited_artifacts"] >= AUDIT_PRESSURE_RATCHET_ARTIFACTS
    assert payload["baseline_inventory"]["summary"]["total_wrappers"] <= (
        BASELINE_WRAPPER_DEBT_BASELINE - 72
    )
    assert payload["baseline_inventory_errors"] == []
    assert payload["audit_pressure"] is None
    assert payload["scope"] == "audited-groups-only"
    assert any(
        record["key"] == "vision.temporal-action-localization.shared-gru"
        for record in payload["records"]
    )
    assert any(record["key"] in RETRIEVAL_AUDITS for record in payload["records"])
