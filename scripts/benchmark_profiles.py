"""Build and verify truthful lesson data/benchmark evidence profiles.

The checked-in files produced here are contracts and attestations, not fresh
benchmark results.  In particular, ``mnist-real-v1`` defines how to execute
the seven MNIST-capable lessons with real data and full default budgets; its
``defined-not-executed`` state must not be presented as a completed run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shlex
import sys
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from scripts.lesson_contracts import (
        OFFLINE_BUILT_IN,
        OFFLINE_EXPLICIT_FAKE,
        discover_lesson_contracts,
    )
    from scripts.lesson_full_run import _benchmark_audit, _inspect_cli
except ModuleNotFoundError:  # Direct ``python scripts/...`` execution.
    from lesson_contracts import (
        OFFLINE_BUILT_IN,
        OFFLINE_EXPLICIT_FAKE,
        discover_lesson_contracts,
    )
    from lesson_full_run import _benchmark_audit, _inspect_cli


SCHEMA_VERSION = 1
PROFILE_ID = "mnist-real-v1"
SEEDS = (41, 42, 43)
CATALOG_PATH = Path("docs/benchmarks/lesson-evidence.json")
PROFILES_PATH = Path("docs/benchmarks/profiles.json")
ATTESTATION_PATH = Path("docs/benchmarks/runtime-attestation.json")
FINAL_RUNTIME_REPORT = Path(
    "outputs/runtime-audit/runs/full-cuda-defaults-339-final-20260830/report.json"
)

CLAIM_TIER = {
    "undocumented": "execution-only",
    "none": "execution-only",
    "disclaimer-only": "execution-only",
    "acceptance-range-only": "acceptance-range",
    "local-offline-benchmark": "local-offline-benchmark",
    "review-required": "blocked-pending-review",
}

REAL_CASES: dict[str, dict[str, Any]] = {
    "generative/lesson_01_vae_mnist": {
        "metrics": ("train_loss", "train_recon", "train_kl", "val_loss"),
        "split": "official MNIST train split; deterministic 90/10 train/validation split",
    },
    "generative/lesson_02_gan_mnist": {
        "metrics": ("d_loss", "g_loss"),
        "split": "official MNIST train split; no validation split",
    },
    "generative/lesson_03_compact_diffusion_mnist": {
        "metrics": ("train_noise_mse", "val_noise_mse"),
        "split": "official MNIST train split; deterministic 90/10 train/validation split",
    },
    "generative/lesson_05_compact_consistency_model": {
        "metrics": ("train_consistency_mse", "val_consistency_mse"),
        "split": "official MNIST train split; deterministic 90/10 train/validation split",
    },
    "vision/lesson_01_mnist_lenet": {
        "metrics": ("train_loss", "train_acc", "eval_loss", "eval_acc"),
        "split": "official MNIST train and test splits",
    },
    "vision/lesson_02_mnist_mlp": {
        "metrics": ("train_loss", "train_acc", "eval_loss", "eval_acc"),
        "split": "official MNIST train and test splits",
    },
    "vision/lesson_03_mnist_alexnet": {
        "metrics": ("train_loss", "train_acc", "eval_loss", "eval_acc"),
        "split": "official MNIST train and test splits",
    },
}

REQUIRED_TRAIN_ARTIFACTS = (
    "checkpoints/checkpoint.pt",
    "config.json",
    "logs/train.log",
    "metrics.jsonl",
)


def repo_root() -> Path:
    return _REPO_ROOT


def _stable_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_digest(root: Path, paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(set(paths), key=lambda value: value.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix().encode()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(_file_sha256(path)))
    return digest.hexdigest()


def _case_source_paths(root: Path, entrypoint: Path) -> list[Path]:
    paths = sorted(entrypoint.parent.glob("*.py"))
    shared = [
        root / "dlhub/data/mnist.py",
        root / "dlhub/data/splits.py",
        root / "dlhub/seed.py",
        root / "dlhub/paths.py",
    ]
    return [path for path in [*paths, *shared] if path.is_file()]


def _dataset_support(defaults: Mapping[str, Any], choices: Mapping[str, list[Any]]) -> bool:
    return defaults.get("--dataset") == "mnist" or "mnist" in choices.get("--dataset", [])


def build_lesson_evidence(root: Path | None = None) -> dict[str, Any]:
    root = (root or repo_root()).resolve()
    lessons: list[dict[str, Any]] = []
    benchmark_counts: Counter[str] = Counter()
    offline_counts: Counter[str] = Counter()
    claim_counts: Counter[str] = Counter()

    for contract in discover_lesson_contracts(root):
        if contract.entrypoint is None or contract.entrypoint_module is None:
            raise ValueError(f"{contract.track}/{contract.lesson}: missing entrypoint")
        lesson_id = f"{contract.track}/{contract.lesson}"
        entrypoint = root / contract.entrypoint
        defaults, choices = _inspect_cli(entrypoint)
        benchmark_class, mentions = _benchmark_audit(entrypoint.parent / "README.md")
        tier = CLAIM_TIER[benchmark_class]
        supports_real = _dataset_support(defaults, choices)
        profile_id = PROFILE_ID if lesson_id in REAL_CASES else None

        if profile_id and not supports_real:
            raise ValueError(f"{lesson_id}: real-data profile does not match the dataset CLI")
        if profile_id and contract.offline_mode != OFFLINE_EXPLICIT_FAKE:
            raise ValueError(f"{lesson_id}: expected an explicit fake-capable offline route")

        if contract.offline_mode == OFFLINE_EXPLICIT_FAKE:
            orchestrated_dataset = "fake"
            data_class = "optional-real-data-with-explicit-fake-offline-route"
        elif contract.offline_mode == OFFLINE_BUILT_IN:
            orchestrated_dataset = "entrypoint-built-in"
            data_class = "built-in-local-or-generated-data"
        else:
            orchestrated_dataset = "external"
            data_class = "external-data-required"

        benchmark_counts[benchmark_class] += 1
        offline_counts[contract.offline_mode] += 1
        claim_counts[tier] += 1
        lessons.append(
            {
                "lesson_id": lesson_id,
                "entrypoint": contract.entrypoint,
                "module": contract.entrypoint_module,
                "offline_mode": contract.offline_mode,
                "data": {
                    "classification": data_class,
                    "declared_dataset_default": defaults.get("--dataset"),
                    "declared_dataset_choices": choices.get("--dataset", []),
                    "full_offline_run_dataset": orchestrated_dataset,
                    "real_data_profile": profile_id,
                    "supports_real_mnist": supports_real,
                    "source_contract": contract.data_source,
                },
                "benchmark": {
                    "classification": benchmark_class,
                    "claim_tier": tier,
                    "mentions": list(mentions),
                    "paper_benchmark_evidence": False,
                },
                "evidence_state": "static-contract-only",
            }
        )

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "policy": {
            "scope": "static data-route and README benchmark-claim inventory for every lesson",
            "does_not_prove": [
                "that a real-data profile has been executed",
                "paper-faithful architecture or hyperparameters",
                "paper benchmark parity or state-of-the-art quality",
            ],
            "runtime_evidence": ATTESTATION_PATH.as_posix(),
        },
        "summary": {
            "lessons": len(lessons),
            "offline_modes": dict(sorted(offline_counts.items())),
            "benchmark_classifications": dict(sorted(benchmark_counts.items())),
            "claim_tiers": dict(sorted(claim_counts.items())),
            "real_data_profile_lessons": sum(
                lesson["data"]["real_data_profile"] is not None for lesson in lessons
            ),
            "paper_benchmark_evidence_lessons": 0,
        },
        "lessons": lessons,
    }
    payload["inventory_sha256"] = _stable_sha256(payload)
    return payload


def _profile_command(module: str) -> list[str]:
    return [
        "{python}",
        "-m",
        module,
        "--dataset",
        "mnist",
        "--seed",
        "{seed}",
        "--device",
        "{device}",
        "--run-name",
        "{run_name}",
    ]


def build_profiles(root: Path | None = None) -> dict[str, Any]:
    root = (root or repo_root()).resolve()
    contracts = {
        f"{contract.track}/{contract.lesson}": contract
        for contract in discover_lesson_contracts(root)
    }
    cases: list[dict[str, Any]] = []
    for lesson_id, case_contract in sorted(REAL_CASES.items()):
        contract = contracts.get(lesson_id)
        if contract is None or contract.entrypoint is None or contract.entrypoint_module is None:
            raise ValueError(f"{lesson_id}: profile lesson or entrypoint is missing")
        entrypoint = root / contract.entrypoint
        defaults, choices = _inspect_cli(entrypoint)
        if not _dataset_support(defaults, choices):
            raise ValueError(f"{lesson_id}: profile requires an MNIST dataset route")
        if any(defaults.get(flag) is not None for flag in ("--max-train-batches", "--max-eval-batches")):
            raise ValueError(f"{lesson_id}: default training limits must be null")
        source_paths = _case_source_paths(root, entrypoint)
        cases.append(
            {
                "lesson_id": lesson_id,
                "entrypoint": contract.entrypoint,
                "module": contract.entrypoint_module,
                "entrypoint_default_dataset": defaults.get("--dataset"),
                "command_template": _profile_command(contract.entrypoint_module),
                "default_budget": {
                    key.removeprefix("--").replace("-", "_"): value
                    for key, value in sorted(defaults.items())
                    if key
                    in {
                        "--batch-size",
                        "--epochs",
                        "--num-diffusion-steps",
                        "--num-discretization-steps",
                        "--num-sample-steps",
                        "--val-fraction",
                    }
                },
                "split_contract": case_contract["split"],
                "required_metrics": list(case_contract["metrics"]),
                "required_artifacts": list(REQUIRED_TRAIN_ARTIFACTS),
                "source_paths": [path.relative_to(root).as_posix() for path in source_paths],
                "implementation_sha256": _source_digest(root, source_paths),
            }
        )

    profile: dict[str, Any] = {
        "profile_id": PROFILE_ID,
        "status": "defined-not-executed",
        "paper_comparable": False,
        "purpose": "real-data execution evidence for the repository's seven MNIST-capable lessons",
        "dataset": {
            "name": "MNIST",
            "kind": "real-public-dataset",
            "provider": "torchvision.datasets.MNIST",
            "official_train_examples": 60_000,
            "official_test_examples": 10_000,
            "download_or_local_cache_required": True,
            "integrity_contract": (
                "record SHA-256 for every local dataset file in the completed report; "
                "torchvision also performs its declared resource-integrity checks"
            ),
        },
        "seeds": list(SEEDS),
        "planned_runs": len(cases) * len(SEEDS),
        "budget_policy": {
            "rule": "use each entrypoint's checked-in defaults",
            "allowed_overrides": ["--dataset", "--device", "--run-name", "--seed"],
            "forbidden_overrides": [
                "--max-*",
                "--batch-size",
                "--epochs",
                "--steps",
                "--num-samples",
            ],
        },
        "report_contract": {
            "required_top_level": [
                "schema_version",
                "profile_id",
                "profile_sha256",
                "status",
                "source_snapshot",
                "environment",
                "dataset_files",
                "runs",
                "summary",
            ],
            "required_source_hashes": ["source_sha256", "git_diff_sha256"],
            "required_run_fields": [
                "lesson_id",
                "seed",
                "state",
                "command",
                "artifacts",
                "metrics",
            ],
            "completion_rule": "all 21 lesson/seed runs pass with hashes, artifacts, and finite required metrics",
            "claim_limit": "a complete report is real-MNIST execution evidence, not a paper benchmark",
        },
        "cases": cases,
    }
    profile["profile_sha256"] = _stable_sha256(profile)
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "truth_policy": (
            "Profiles describe planned reproducible executions. Only a validated report may change "
            "an execution claim from planned to observed."
        ),
        "profiles": [profile],
    }
    payload["inventory_sha256"] = _stable_sha256(payload)
    return payload


def _runtime_attestation(report_path: Path) -> dict[str, Any]:
    report_bytes = report_path.read_bytes()
    report = json.loads(report_bytes)
    lessons = report["lessons"]
    commands = [entry["command"] for entry in lessons.values()]
    train_entries = [entry for entry in lessons.values() if entry["artifact_validation"]["required"]]
    metric_records = sum(
        entry["artifact_validation"].get("metric_records", 0) for entry in train_entries
    )
    positive_gpu = sum(entry["gpu_memory"].get("peak_bytes", 0) > 0 for entry in train_entries)
    elapsed = sum(float(entry["elapsed_seconds"]) for entry in lessons.values())
    injected_fake = sum(
        any(command[index : index + 2] == ["--dataset", "fake"] for index in range(len(command) - 1))
        for command in commands
    )
    forbidden_limit_flags = sum(
        any(token.startswith("--max-") for token in command) for command in commands
    )

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "evidence_kind": "historical-complete-default-offline-runtime-attestation",
        "report": {
            "path": FINAL_RUNTIME_REPORT.as_posix(),
            "sha256": hashlib.sha256(report_bytes).hexdigest(),
            "created_at": report["created_at"],
            "updated_at": report["updated_at"],
            "status": report["status"],
        },
        "observed": {
            "lessons": len(lessons),
            "passed": sum(entry["state"] == "passed" for entry in lessons.values()),
            "single_attempt_lessons": sum(entry["attempts"] == 1 for entry in lessons.values()),
            "elapsed_seconds_sum": round(elapsed, 6),
            "validated_train_artifacts": sum(
                entry["artifact_validation"].get("ok") is True for entry in train_entries
            ),
            "metric_records": metric_records,
            "positive_cuda_peak_train_lessons": positive_gpu,
            "commands_selecting_fake": injected_fake,
            "commands_with_max_limit_flags": forbidden_limit_flags,
        },
        "execution": {
            "device": report["config"]["device"],
            "device_details": report["config"]["device_details"],
            "network_policy": report["config"]["network_policy"],
            "serial": report["config"]["serial"],
        },
        "source_snapshot": {
            key: report["source_snapshot"][key]
            for key in (
                "source_sha256",
                "source_file_count",
                "source_bytes",
                "git_head",
                "git_diff_sha256",
                "git_status_sha256",
            )
        },
        "inventory_sha256": report["inventory_sha256"],
        "claim_boundary": {
            "real_data_executed": False,
            "paper_benchmark_evidence": False,
            "proves": "default-budget offline lesson execution and artifact contracts on the recorded Linux/CUDA snapshot",
            "does_not_prove": [
                "real-dataset training",
                "paper benchmark parity",
                "paper-faithful model implementations",
                "Windows or MPS full-curriculum execution",
            ],
        },
    }
    payload["attestation_sha256"] = _stable_sha256(payload)
    return payload


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _profile_by_id(profiles: Mapping[str, Any], profile_id: str) -> dict[str, Any]:
    for profile in profiles.get("profiles", []):
        if profile.get("profile_id") == profile_id:
            return profile
    raise ValueError(f"unknown profile: {profile_id}")


def validate_report(report: Mapping[str, Any], profile: Mapping[str, Any]) -> list[str]:
    """Validate a real-data report without upgrading its scientific claim."""

    errors: list[str] = []
    required_top = set(profile["report_contract"]["required_top_level"])
    missing_top = sorted(required_top - report.keys())
    if missing_top:
        errors.append(f"missing top-level fields: {missing_top}")
        return errors
    if report.get("profile_id") != profile["profile_id"]:
        errors.append("profile_id does not match")
    if report.get("profile_sha256") != profile["profile_sha256"]:
        errors.append("profile_sha256 does not match the checked profile")
    if report.get("paper_benchmark_evidence") not in (None, False):
        errors.append("MNIST profile reports cannot claim paper benchmark evidence")

    snapshot = report.get("source_snapshot")
    if not isinstance(snapshot, Mapping):
        errors.append("source_snapshot must be an object")
    else:
        for key in profile["report_contract"]["required_source_hashes"]:
            value = snapshot.get(key)
            if not isinstance(value, str) or len(value) != 64:
                errors.append(f"source_snapshot.{key} must be a SHA-256 string")

    dataset_files = report.get("dataset_files")
    if not isinstance(dataset_files, list) or not dataset_files:
        errors.append("dataset_files must contain hashed local MNIST files")
    else:
        for index, item in enumerate(dataset_files):
            if not isinstance(item, Mapping) or not isinstance(item.get("sha256"), str):
                errors.append(f"dataset_files[{index}] is missing sha256")

    expected = {
        (case["lesson_id"], seed)
        for case in profile["cases"]
        for seed in profile["seeds"]
    }
    cases = {case["lesson_id"]: case for case in profile["cases"]}
    observed: set[tuple[str, int]] = set()
    runs = report.get("runs")
    if not isinstance(runs, list):
        errors.append("runs must be a list")
        return errors
    for index, run in enumerate(runs):
        if not isinstance(run, Mapping):
            errors.append(f"runs[{index}] must be an object")
            continue
        missing = set(profile["report_contract"]["required_run_fields"]) - run.keys()
        if missing:
            errors.append(f"runs[{index}] missing fields: {sorted(missing)}")
            continue
        key = (run["lesson_id"], run["seed"])
        if key in observed:
            errors.append(f"duplicate run: {key}")
        observed.add(key)
        case = cases.get(run["lesson_id"])
        if case is None or run["seed"] not in profile["seeds"]:
            errors.append(f"unexpected run: {key}")
            continue
        if run["state"] != "passed":
            errors.append(f"run did not pass: {key}")
        command = run["command"]
        if not isinstance(command, list) or not all(isinstance(token, str) for token in command):
            errors.append(f"run command is invalid: {key}")
        else:
            joined = " ".join(command)
            if "--dataset mnist" not in joined or f"--seed {run['seed']}" not in joined:
                errors.append(f"run command does not select MNIST and its seed: {key}")
            if any(token.startswith("--max-") for token in command):
                errors.append(f"run command contains a max-* limit: {key}")
            forbidden = set(profile["budget_policy"]["forbidden_overrides"][1:])
            if forbidden.intersection(command):
                errors.append(f"run command overrides a default budget: {key}")
        artifacts = run["artifacts"]
        artifact_paths = {
            item.get("path")
            for item in artifacts
            if isinstance(item, Mapping) and isinstance(item.get("sha256"), str)
        }
        if not set(case["required_artifacts"]).issubset(artifact_paths):
            errors.append(f"run is missing hashed artifacts: {key}")
        metrics = run["metrics"]
        if not isinstance(metrics, list) or not metrics:
            errors.append(f"run has no metrics: {key}")
        else:
            final = metrics[-1]
            if not isinstance(final, Mapping):
                errors.append(f"run final metrics are invalid: {key}")
            else:
                for metric in case["required_metrics"]:
                    value = final.get(metric)
                    if (
                        isinstance(value, bool)
                        or not isinstance(value, int | float)
                        or not math.isfinite(value)
                    ):
                        errors.append(f"run final metric {metric} is not finite: {key}")

    if observed != expected:
        errors.append(
            f"run matrix mismatch: expected {len(expected)}, observed {len(observed)}"
        )
    summary = report.get("summary")
    if not isinstance(summary, Mapping) or summary.get("passed") != len(expected):
        errors.append("summary.passed does not cover the complete run matrix")
    if report.get("status") != "complete":
        errors.append("report status is not complete")
    return errors


def _json_text(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    from dlhub._atomic import atomic_write

    data = _json_text(payload).encode()
    path.parent.mkdir(parents=True, exist_ok=True)

    def write(stream: Any) -> None:
        stream.write(data)

    atomic_write(path, write)


def _check_file(path: Path, expected: Mapping[str, Any]) -> str | None:
    if not path.is_file():
        return f"missing: {path}"
    try:
        actual = _load_json(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return f"invalid: {path}: {exc}"
    if actual != expected:
        return f"stale: {path}"
    return None


def _validate_attestation(attestation: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    body = dict(attestation)
    digest = body.pop("attestation_sha256", None)
    if digest != _stable_sha256(body):
        errors.append("runtime attestation content hash does not match")
    if attestation.get("claim_boundary", {}).get("real_data_executed") is not False:
        errors.append("offline runtime attestation must not claim real-data execution")
    if attestation.get("claim_boundary", {}).get("paper_benchmark_evidence") is not False:
        errors.append("offline runtime attestation must not claim a paper benchmark")
    if attestation.get("observed", {}).get("commands_with_max_limit_flags") != 0:
        errors.append("offline runtime attestation contains limited training commands")
    return errors


def _render_commands(profile: Mapping[str, Any], *, python: str, device: str) -> list[str]:
    commands: list[str] = []
    for case in profile["cases"]:
        short = case["lesson_id"].replace("/", "-").replace("_", "-")
        for seed in profile["seeds"]:
            replacements = {
                "{python}": python,
                "{device}": device,
                "{run_name}": f"{profile['profile_id']}-{short}-s{seed}",
                "{seed}": str(seed),
            }
            command = [replacements.get(token, token) for token in case["command_template"]]
            commands.append(shlex.join(command))
    return commands


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--check", action="store_true", help="verify checked-in evidence files")
    action.add_argument("--write", action="store_true", help="regenerate checked-in evidence files")
    action.add_argument("--list", action="store_true", help="list available profiles")
    action.add_argument("--commands", metavar="PROFILE", help="print the full planned run matrix")
    action.add_argument("--validate-report", type=Path, metavar="PATH")
    parser.add_argument("--device", default="cuda", help="device used by --commands")
    parser.add_argument("--python", default=sys.executable, help="Python used by --commands")
    args = parser.parse_args(argv)

    root = repo_root()
    catalog = build_lesson_evidence(root)
    profiles = build_profiles(root)
    profile = _profile_by_id(profiles, PROFILE_ID)

    if args.list:
        for item in profiles["profiles"]:
            print(
                f"{item['profile_id']}: {len(item['cases'])} lessons x "
                f"{len(item['seeds'])} seeds = {item['planned_runs']} planned runs "
                f"[{item['status']}]"
            )
        return 0
    if args.commands:
        selected = _profile_by_id(profiles, args.commands)
        print("\n".join(_render_commands(selected, python=args.python, device=args.device)))
        return 0
    if args.validate_report:
        try:
            report = _load_json(args.validate_report)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            parser.error(str(exc))
        errors = validate_report(report, profile)
        if errors:
            print("real-data report: INVALID")
            for error in errors:
                print(f"- {error}")
            return 1
        print(f"real-data report: OK ({profile['planned_runs']} observed runs)")
        return 0

    paths = {
        root / CATALOG_PATH: catalog,
        root / PROFILES_PATH: profiles,
    }
    report_path = root / FINAL_RUNTIME_REPORT
    attestation_path = root / ATTESTATION_PATH
    if args.write:
        if not report_path.is_file():
            parser.error(
                f"cannot regenerate {ATTESTATION_PATH}: source report is absent: {report_path}"
            )
        paths[attestation_path] = _runtime_attestation(report_path)
        for path, payload in paths.items():
            _write_json(path, payload)
        print(
            f"benchmark evidence: wrote {len(catalog['lessons'])} lessons, "
            f"{profile['planned_runs']} planned real-data runs, and one runtime attestation"
        )
        return 0

    failures = [error for path, payload in paths.items() if (error := _check_file(path, payload))]
    if not attestation_path.is_file():
        failures.append(f"missing: {attestation_path}")
    else:
        try:
            checked_attestation = _load_json(attestation_path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            failures.append(f"invalid: {attestation_path}: {exc}")
        else:
            failures.extend(_validate_attestation(checked_attestation))
            if report_path.is_file():
                expected_attestation = _runtime_attestation(report_path)
                if checked_attestation != expected_attestation:
                    failures.append(f"stale: {attestation_path}")
    if failures:
        print("benchmark evidence: FAILED")
        for failure in failures:
            print(f"- {failure}")
        print("Run: python scripts/benchmark_profiles.py --write")
        return 1
    print(
        f"benchmark evidence: OK ({len(catalog['lessons'])} lessons; "
        f"{profile['planned_runs']} planned real-data runs; no paper benchmark claim)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
