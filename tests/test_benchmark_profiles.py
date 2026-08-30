from __future__ import annotations

import copy
import json
from typing import Any

import pytest

import scripts.benchmark_profiles as benchmark_profiles


@pytest.fixture(scope="module")
def profiles() -> dict[str, Any]:
    return benchmark_profiles.build_profiles()


@pytest.fixture(scope="module")
def profile(profiles: dict[str, Any]) -> dict[str, Any]:
    return profiles["profiles"][0]


def _valid_report(profile: dict[str, Any]) -> dict[str, Any]:
    runs = []
    for case in profile["cases"]:
        for seed in profile["seeds"]:
            replacements = {
                "{python}": "python",
                "{device}": "cuda",
                "{run_name}": f"report-{seed}",
                "{seed}": str(seed),
            }
            command = [replacements.get(token, token) for token in case["command_template"]]
            runs.append(
                {
                    "lesson_id": case["lesson_id"],
                    "seed": seed,
                    "state": "passed",
                    "command": command,
                    "artifacts": [
                        {"path": path, "sha256": "a" * 64, "size": 1}
                        for path in case["required_artifacts"]
                    ],
                    "metrics": [
                        {name: float(index + 1) for index, name in enumerate(case["required_metrics"])}
                    ],
                }
            )
    return {
        "schema_version": 1,
        "profile_id": profile["profile_id"],
        "profile_sha256": profile["profile_sha256"],
        "status": "complete",
        "paper_benchmark_evidence": False,
        "source_snapshot": {
            "source_sha256": "b" * 64,
            "git_diff_sha256": "c" * 64,
        },
        "environment": {"python": "3.10", "device": "cuda"},
        "dataset_files": [{"path": "MNIST/raw/train-images", "sha256": "d" * 64}],
        "runs": runs,
        "summary": {"passed": len(runs), "total": len(runs)},
    }


def test_lesson_evidence_covers_every_lesson_without_paper_claims() -> None:
    catalog = benchmark_profiles.build_lesson_evidence()

    assert catalog["summary"]["lessons"] == 339
    assert catalog["summary"]["offline_modes"] == {"built-in": 332, "explicit-fake": 7}
    assert catalog["summary"]["real_data_profile_lessons"] == 7
    assert catalog["summary"]["paper_benchmark_evidence_lessons"] == 0
    assert len(catalog["lessons"]) == 339
    assert all(
        lesson["benchmark"]["paper_benchmark_evidence"] is False
        for lesson in catalog["lessons"]
    )
    assert all(lesson["evidence_state"] == "static-contract-only" for lesson in catalog["lessons"])


def test_mnist_profile_is_real_data_full_budget_but_not_an_executed_claim(
    profile: dict[str, Any],
) -> None:
    assert profile["profile_id"] == "mnist-real-v1"
    assert profile["status"] == "defined-not-executed"
    assert profile["paper_comparable"] is False
    assert profile["dataset"]["kind"] == "real-public-dataset"
    assert len(profile["cases"]) == 7
    assert profile["seeds"] == [41, 42, 43]
    assert profile["planned_runs"] == 21

    expected_epochs = {
        "generative/lesson_01_vae_mnist": 5,
        "generative/lesson_02_gan_mnist": 10,
        "generative/lesson_03_compact_diffusion_mnist": 5,
        "generative/lesson_05_compact_consistency_model": 5,
        "vision/lesson_01_mnist_lenet": 1,
        "vision/lesson_02_mnist_mlp": 1,
        "vision/lesson_03_mnist_alexnet": 1,
    }
    for case in profile["cases"]:
        command = case["command_template"]
        assert command[command.index("--dataset") + 1] == "mnist"
        assert not any(token.startswith("--max-") for token in command)
        assert "--epochs" not in command
        assert "--batch-size" not in command
        assert case["default_budget"]["epochs"] == expected_epochs[case["lesson_id"]]
        assert len(case["implementation_sha256"]) == 64


def test_checked_evidence_files_match_source(profiles: dict[str, Any]) -> None:
    root = benchmark_profiles.repo_root()
    expected = {
        benchmark_profiles.CATALOG_PATH: benchmark_profiles.build_lesson_evidence(root),
        benchmark_profiles.PROFILES_PATH: profiles,
    }
    for relative, payload in expected.items():
        assert json.loads((root / relative).read_text(encoding="utf-8")) == payload

    attestation = json.loads((root / benchmark_profiles.ATTESTATION_PATH).read_text())
    assert benchmark_profiles._validate_attestation(attestation) == []
    assert attestation["observed"]["lessons"] == 339
    assert attestation["observed"]["passed"] == 339
    assert attestation["observed"]["commands_selecting_fake"] == 7
    assert attestation["claim_boundary"]["real_data_executed"] is False


def test_real_data_report_validator_accepts_only_complete_hashed_matrix(
    profile: dict[str, Any],
) -> None:
    report = _valid_report(profile)
    assert benchmark_profiles.validate_report(report, profile) == []

    incomplete = copy.deepcopy(report)
    incomplete["runs"].pop()
    incomplete["summary"]["passed"] -= 1
    errors = benchmark_profiles.validate_report(incomplete, profile)
    assert any("run matrix mismatch" in error for error in errors)

    overclaim = copy.deepcopy(report)
    overclaim["paper_benchmark_evidence"] = True
    errors = benchmark_profiles.validate_report(overclaim, profile)
    assert any("cannot claim paper benchmark" in error for error in errors)

    limited = copy.deepcopy(report)
    limited["runs"][0]["command"].extend(["--max-train-batches", "1"])
    errors = benchmark_profiles.validate_report(limited, profile)
    assert any("max-* limit" in error for error in errors)
