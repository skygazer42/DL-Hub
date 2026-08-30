from __future__ import annotations

import ast
import json
import math
import os
import shutil
import subprocess
import sys
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

import scripts.lesson_full_run as full_run

torch = pytest.importorskip("torch")


STANDARD_TRAIN_ARTIFACTS = {
    "config.json",
    "metrics.jsonl",
    "logs/train.log",
    "checkpoints/checkpoint.pt",
}


@pytest.fixture(scope="module")
def lesson_manifest() -> dict[str, Any]:
    return full_run.build_lesson_manifest()


@pytest.fixture
def audit_test_root() -> Iterator[Path]:
    parent = full_run.default_runtime_root() / "pytest"
    path = parent / f"{os.getpid()}-{uuid.uuid4().hex}"
    path.mkdir(parents=True)
    try:
        yield path
    finally:
        assert path.parent == parent
        shutil.rmtree(path)


def test_manifest_covers_complete_default_budget_inventory(
    lesson_manifest: dict[str, Any],
) -> None:
    summary = lesson_manifest["summary"]

    assert summary["lessons"] == 339
    assert summary["train_entrypoints"] == 338
    assert summary["run_entrypoints"] == 1
    assert summary["built_in_offline"] == 332
    assert summary["explicit_fake"] == 7
    assert summary["external_only"] == 0
    assert summary["estimated_lessons"] == 319
    assert summary["unestimated_lessons"] == 20
    assert summary["estimated_train_batches"] == 71_360
    assert summary["maximum_estimated_train_batches"] == 3_200
    assert (
        summary["maximum_estimated_lesson"]
        == "pointcloud/lesson_10_pointcloud_selfsupervised_pointmae"
    )
    assert summary["epoch_default_lessons"] == 336
    assert summary["training_limit_flag_lessons"] == 329
    assert summary["non_null_training_limit_defaults"] == 0
    assert summary["standard_artifact_train_lessons"] == 338
    assert summary["benchmark_review_required"] == 0

    snapshot = lesson_manifest["source_snapshot"]
    assert len(snapshot["source_sha256"]) == 64
    assert len(snapshot["git_diff_sha256"]) == 64
    assert snapshot["source_file_count"] >= 339
    assert snapshot["source_bytes"] > 0
    assert lesson_manifest["inventory_sha256"]


def test_manifest_never_hides_a_training_budget_override(
    lesson_manifest: dict[str, Any],
) -> None:
    fake_commands = 0
    for spec in lesson_manifest["lessons"]:
        assert all(value is None for value in spec["limit_defaults"].values())
        command = full_run.build_full_command(
            spec,
            python=sys.executable,
            device="cpu",
            run_name="audit",
        )
        flags = {token for token in command if token.startswith("--")}
        assert not any(flag.startswith("--max-") for flag in flags)
        assert not flags.intersection(full_run.TRAINING_BUDGET_FLAGS)
        assert not flags.intersection(full_run.DATA_SCALE_FLAGS)
        assert flags <= full_run.ALLOWED_INJECTED_FLAGS
        if spec["offline_mode"] == full_run.OFFLINE_EXPLICIT_FAKE:
            assert command[-2:] == ["--dataset", "fake"]
            fake_commands += 1
        else:
            assert "--dataset" not in command

    assert fake_commands == 7


def test_attempt_run_names_are_unique_for_all_lessons(
    lesson_manifest: dict[str, Any],
) -> None:
    names = [
        full_run._attempt_run_name("full-audit", spec["lesson_id"], 1)
        for spec in lesson_manifest["lessons"]
    ]

    assert len(names) == 339
    assert len(set(names)) == 339
    assert all(name.endswith("-a001") for name in names)


def test_train_output_identity_matches_its_contract(lesson_manifest: dict[str, Any]) -> None:
    root = full_run.repo_root()
    for spec in lesson_manifest["lessons"]:
        if spec["kind"] != "train":
            continue
        path = root / spec["entrypoint"]
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        identities: list[tuple[str, str]] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = node.func.id if isinstance(node.func, ast.Name) else None
            if name != "build_run_paths":
                continue
            keywords = {keyword.arg: keyword.value for keyword in node.keywords}
            track = ast.literal_eval(keywords["track"])
            lesson = ast.literal_eval(keywords["lesson"])
            identities.append((track, lesson))
        assert identities == [(spec["track"], spec["lesson"])], spec["lesson_id"]


@pytest.mark.parametrize(
    ("relative", "expected_identity", "foreign_identity", "expected_logger"),
    [
        (
            "tracks/llm/lesson_05_compact_prefix_tuning/train.py",
            "lesson_05_compact_prefix_tuning",
            "lesson_02_compact_chat_sft",
            "llm.compact_prefix_tuning",
        ),
        (
            "tracks/multimodal/lesson_19_audio_text_understanding/train.py",
            "lesson_19_audio_text_understanding",
            "lesson_01_clip_compact_retrieval",
            "multimodal.audio_text_understanding",
        ),
    ],
)
def test_regression_for_copied_lesson_identity(
    relative: str,
    expected_identity: str,
    foreign_identity: str,
    expected_logger: str,
) -> None:
    text = (full_run.repo_root() / relative).read_text(encoding="utf-8")

    assert foreign_identity not in text
    assert text.count(expected_identity) >= 3
    assert expected_logger in text


def test_every_train_requires_nonempty_standard_artifacts(
    lesson_manifest: dict[str, Any],
) -> None:
    train_specs = [spec for spec in lesson_manifest["lessons"] if spec["kind"] == "train"]
    run_specs = [spec for spec in lesson_manifest["lessons"] if spec["kind"] == "run"]

    assert len(train_specs) == 338
    assert all(set(spec["required_artifacts"]) == STANDARD_TRAIN_ARTIFACTS for spec in train_specs)
    assert len(run_specs) == 1
    assert run_specs[0]["required_artifacts"] == ()


def test_benchmark_mentions_are_resolved_without_discarding_evidence(
    lesson_manifest: dict[str, Any],
) -> None:
    classified = {
        spec["lesson_id"]: spec for spec in lesson_manifest["lessons"] if spec["benchmark_mentions"]
    }

    assert (
        classified["vision/lesson_26_synthetic_text_detection"]["benchmark_classification"]
        == "acceptance-range-only"
    )
    assert (
        classified["vision/lesson_70_synthetic_video_understanding"]["benchmark_classification"]
        == "acceptance-range-only"
    )
    assert (
        classified["vision/lesson_87_synthetic_action_recognition"]["benchmark_classification"]
        == "acceptance-range-only"
    )
    assert (
        classified["vision/lesson_72_synthetic_video_enhancement"]["benchmark_classification"]
        == "local-offline-benchmark"
    )


@pytest.mark.parametrize("cpu_threads", [0, -1, 1.5, True])
def test_runner_rejects_invalid_cpu_threads(cpu_threads: object) -> None:
    with pytest.raises(ValueError, match="cpu_threads must be positive"):
        full_run.run_manifest_selection(
            {},
            [],
            root=full_run.repo_root(),
            runtime_root=full_run.default_runtime_root(),
            report_path=full_run.default_runtime_root() / "unused.json",
            run_id="unused",
            device="cpu",
            device_details={},
            cuda_index=0,
            cpu_threads=cpu_threads,  # type: ignore[arg-type]
            timeout_seconds=1,
            cuda_lock_timeout_seconds=1,
            allow_network=False,
            resume=False,
            retry_failed=False,
        )


@pytest.mark.parametrize("timeout", [0, -1, math.nan, math.inf, -math.inf, True])
def test_runner_rejects_invalid_lesson_timeouts(timeout: float) -> None:
    with pytest.raises(ValueError, match="timeout must be positive and finite"):
        full_run.run_manifest_selection(
            {},
            [],
            root=full_run.repo_root(),
            runtime_root=full_run.default_runtime_root(),
            report_path=full_run.default_runtime_root() / "unused.json",
            run_id="unused",
            device="cpu",
            device_details={},
            cuda_index=0,
            cpu_threads=1,
            timeout_seconds=timeout,
            cuda_lock_timeout_seconds=1,
            allow_network=False,
            resume=False,
            retry_failed=False,
        )


@pytest.mark.parametrize("timeout", [0, -1, math.nan, math.inf, -math.inf, True])
def test_runner_rejects_invalid_cuda_lock_timeouts(timeout: float) -> None:
    with pytest.raises(ValueError, match="CUDA lock timeout must be positive and finite"):
        full_run.run_manifest_selection(
            {},
            [],
            root=full_run.repo_root(),
            runtime_root=full_run.default_runtime_root(),
            report_path=full_run.default_runtime_root() / "unused.json",
            run_id="unused",
            device="cpu",
            device_details={},
            cuda_index=0,
            cpu_threads=1,
            timeout_seconds=1,
            cuda_lock_timeout_seconds=timeout,
            allow_network=False,
            resume=False,
            retry_failed=False,
        )


def test_selection_is_explicit_and_deterministic(lesson_manifest: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="execution is never implicit"):
        full_run._select_lessons(
            lesson_manifest,
            run_all=False,
            lessons=[],
            tracks=[],
            limit=None,
        )
    with pytest.raises(ValueError, match="unknown lessons"):
        full_run._select_lessons(
            lesson_manifest,
            run_all=False,
            lessons=["missing/lesson"],
            tracks=[],
            limit=None,
        )

    first = full_run._select_lessons(
        lesson_manifest,
        run_all=True,
        lessons=[],
        tracks=[],
        limit=3,
    )
    assert [spec["lesson_id"] for spec in first] == [
        spec["lesson_id"] for spec in lesson_manifest["lessons"][:3]
    ]


def test_source_snapshot_detects_content_drift(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "source.py"
    source.write_text("value = 1\n", encoding="utf-8")
    monkeypatch.setattr(full_run, "SOURCE_PATHS", ("source.py",))
    snapshot = full_run._source_snapshot(tmp_path)

    full_run._assert_source_snapshot(tmp_path, snapshot)
    source.write_text("value = 2\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="runtime source changed"):
        full_run._assert_source_snapshot(tmp_path, snapshot)


def test_runtime_environment_is_fully_under_configured_runtime_root(
    audit_test_root: Path,
) -> None:
    case_root = audit_test_root / "case"
    bootstrap = full_run._prepare_bootstrap(audit_test_root, allow_network=False)
    with full_run._short_tmpdir(case_root) as tmpdir:
        environment = full_run._runtime_environment(
            root=full_run.repo_root(),
            run_root=audit_test_root,
            case_root=case_root,
            tmpdir=tmpdir,
            bootstrap=bootstrap,
            allow_network=False,
            device="cpu",
            cuda_index=0,
            cpu_threads=1,
        )

        assert Path(environment["TMPDIR"]) == tmpdir
        assert tmpdir.parent == Path("/tmp")
        assert tmpdir.is_symlink()
        assert tmpdir.resolve() == (case_root / "tmp").resolve()

    assert audit_test_root.resolve().is_relative_to(full_run.default_runtime_root().resolve())
    for variable in (
        "DLHUB_OUTPUTS_DIR",
        "DLHUB_RUNTIME_AUDIT_GPU_METRICS_DIR",
        "HF_HOME",
        "MPLCONFIGDIR",
        "TORCH_HOME",
        "XDG_CACHE_HOME",
    ):
        assert Path(environment[variable]).is_relative_to(audit_test_root)
    assert not os.path.lexists(environment["TMPDIR"])
    assert environment["HF_HUB_OFFLINE"] == "1"
    assert environment["CUDA_VISIBLE_DEVICES"] == ""


def test_offline_bootstrap_blocks_python_inet_sockets(audit_test_root: Path) -> None:
    case_root = audit_test_root / "socket-case"
    for path in (case_root / "tmp", case_root / "gpu-metrics"):
        path.mkdir(parents=True)
    bootstrap = full_run._prepare_bootstrap(audit_test_root, allow_network=False)
    with full_run._short_tmpdir(case_root) as tmpdir:
        environment = full_run._runtime_environment(
            root=full_run.repo_root(),
            run_root=audit_test_root,
            case_root=case_root,
            tmpdir=tmpdir,
            bootstrap=bootstrap,
            allow_network=False,
            device="cpu",
            cuda_index=0,
            cpu_threads=1,
        )
        process = subprocess.run(
            [
                sys.executable,
                "-c",
                ("import socket; socket.create_connection(('127.0.0.1', 9), timeout=0.01)"),
            ],
            cwd=case_root,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )

    assert process.returncode != 0
    assert "runtime audit blocked socket.create_connection" in process.stderr


def test_artifact_validator_requires_safe_finite_standard_outputs(
    audit_test_root: Path,
) -> None:
    run_dir = audit_test_root / "outputs" / "track" / "lesson" / "run"
    (run_dir / "logs").mkdir(parents=True)
    (run_dir / "checkpoints").mkdir()
    (run_dir / "config.json").write_text('{"seed": 0}\n', encoding="utf-8")
    (run_dir / "metrics.jsonl").write_text('{"loss": 1.0}\n', encoding="utf-8")
    (run_dir / "logs" / "train.log").write_text("finished\n", encoding="utf-8")
    torch.save(
        {"model_state": {"weight": torch.ones(1)}},
        run_dir / "checkpoints" / "checkpoint.pt",
    )
    spec = {"kind": "train", "required_artifacts": sorted(STANDARD_TRAIN_ARTIFACTS)}

    validation = full_run._validate_artifacts(spec, run_dir)
    assert validation["ok"] is True
    assert validation["metric_records"] == 1
    assert validation["checkpoint_keys"] == ["model_state"]

    (run_dir / "metrics.jsonl").write_text('{"loss": NaN}\n', encoding="utf-8")
    validation = full_run._validate_artifacts(spec, run_dir)
    assert validation["ok"] is False
    assert any("non-finite" in error for error in validation["errors"])


@pytest.mark.parametrize("marker", [None, False, 1, "true"])
def test_artifact_validator_only_accepts_explicit_true_model_free_marker(
    audit_test_root: Path,
    marker: object,
) -> None:
    run_dir = audit_test_root / "model-free" / str(marker)
    (run_dir / "logs").mkdir(parents=True)
    (run_dir / "checkpoints").mkdir()
    (run_dir / "config.json").write_text("{}\n", encoding="utf-8")
    (run_dir / "metrics.jsonl").write_text('{"accuracy": 1.0}\n', encoding="utf-8")
    (run_dir / "logs" / "train.log").write_text("finished\n", encoding="utf-8")
    extra = {} if marker is None else {"model_free": marker}
    torch.save(
        {"model_state": {}, "epoch": 1, "extra": extra},
        run_dir / "checkpoints" / "checkpoint.pt",
    )
    spec = {"kind": "train", "required_artifacts": sorted(STANDARD_TRAIN_ARTIFACTS)}

    validation = full_run._validate_artifacts(spec, run_dir)

    assert validation["ok"] is False
    assert validation["model_free"] is False
    assert validation["errors"] == ["checkpoint.pt has no non-empty model_state"]


def test_verified_model_free_checkpoint_exempts_only_cuda_allocation(
    audit_test_root: Path,
) -> None:
    run_dir = audit_test_root / "verified-model-free"
    (run_dir / "logs").mkdir(parents=True)
    (run_dir / "checkpoints").mkdir()
    (run_dir / "config.json").write_text("{}\n", encoding="utf-8")
    (run_dir / "metrics.jsonl").write_text('{"accuracy": 1.0}\n', encoding="utf-8")
    (run_dir / "logs" / "train.log").write_text("finished\n", encoding="utf-8")
    torch.save(
        {"model_state": {}, "epoch": 1, "extra": {"model_free": True}},
        run_dir / "checkpoints" / "checkpoint.pt",
    )
    spec = {"kind": "train", "required_artifacts": sorted(STANDARD_TRAIN_ARTIFACTS)}
    validation = full_run._validate_artifacts(spec, run_dir)

    assert validation["ok"] is True
    assert validation["model_free"] is True
    assert (
        full_run._gpu_measurement_error(
            spec,
            device="cuda",
            returncode=0,
            gpu_memory={"available": True, "peak_bytes": 0},
            artifact_validation=validation,
        )
        is None
    )


def test_cuda_train_requires_peak_memory_but_cpu_and_run_do_not() -> None:
    missing = {"available": False, "peak_bytes": None}
    measured = {"available": True, "peak_bytes": 1}

    assert full_run._gpu_measurement_error(
        {"kind": "train"}, device="cuda", returncode=0, gpu_memory=missing
    )
    assert (
        full_run._gpu_measurement_error(
            {"kind": "train"}, device="cuda", returncode=0, gpu_memory=measured
        )
        is None
    )
    assert (
        full_run._gpu_measurement_error(
            {"kind": "train"}, device="cpu", returncode=0, gpu_memory=missing
        )
        is None
    )
    assert (
        full_run._gpu_measurement_error(
            {"kind": "run"}, device="cuda", returncode=0, gpu_memory=missing
        )
        is None
    )


@pytest.mark.parametrize("peak_bytes", [0, -1, 1.5, True, None, "1"])
def test_cuda_train_rejects_nonpositive_or_noninteger_peak_memory(
    peak_bytes: object,
) -> None:
    error = full_run._gpu_measurement_error(
        {"kind": "train"},
        device="cuda",
        returncode=0,
        gpu_memory={"available": True, "peak_bytes": peak_bytes},
    )

    assert error == "missing positive integer child PyTorch CUDA peak-memory measurement"


def test_run_one_enforces_timeout_inside_isolated_case_root(
    monkeypatch, audit_test_root: Path
) -> None:
    monkeypatch.setattr(
        full_run,
        "build_full_command",
        lambda *args, **kwargs: [sys.executable, "-c", "import time; time.sleep(60)"],
    )
    bootstrap = full_run._prepare_bootstrap(audit_test_root, allow_network=False)
    result = full_run._run_one(
        {"lesson_id": "demo/lesson", "track": "demo", "lesson": "lesson", "kind": "run"},
        root=full_run.repo_root(),
        run_root=audit_test_root,
        run_name="full-timeout-a001",
        attempt=1,
        device="cpu",
        cuda_index=0,
        cpu_threads=1,
        timeout_seconds=0.05,
        bootstrap=bootstrap,
        allow_network=False,
    )

    assert result["state"] == "timed_out"
    assert result["elapsed_seconds"] < 10
    assert result["case_root"].startswith(str(audit_test_root))
    assert result["case_root"].endswith("demo__lesson/attempt-001")


def test_run_one_uses_short_tmpdir_for_long_paths_and_dataloader_workers(
    monkeypatch, audit_test_root: Path
) -> None:
    long_root = audit_test_root / "runs" / ("long-run-id-" + "x" * 96)
    expected_runtime_root = str(long_root.resolve())
    code = f"""
import os
import torch
from torch.utils.data import DataLoader, TensorDataset

loader = DataLoader(TensorDataset(torch.arange(64)), batch_size=8, num_workers=2)
assert sum(int(batch[0].sum()) for batch in loader) == sum(range(64))
tmpdir = os.environ["TMPDIR"]
assert tmpdir.startswith("/tmp/dlh-")
assert os.path.commonpath([os.path.realpath(tmpdir), {expected_runtime_root!r}]) == {expected_runtime_root!r}
print(tmpdir)
"""
    monkeypatch.setattr(
        full_run,
        "build_full_command",
        lambda *args, **kwargs: [sys.executable, "-c", code],
    )
    bootstrap = full_run._prepare_bootstrap(long_root, allow_network=False)
    result = full_run._run_one(
        {
            "lesson_id": "demo/lesson_" + "y" * 80,
            "track": "demo",
            "lesson": "lesson_" + "y" * 80,
            "kind": "run",
        },
        root=full_run.repo_root(),
        run_root=long_root,
        run_name="full-" + "z" * 100,
        attempt=1,
        device="cpu",
        cuda_index=0,
        cpu_threads=1,
        timeout_seconds=30,
        bootstrap=bootstrap,
        allow_network=False,
    )

    assert result["state"] == "passed", result["stderr_tail"]
    short_tmpdir = result["stdout_tail"].strip().splitlines()[-1]
    assert short_tmpdir.startswith("/tmp/dlh-")
    assert not os.path.lexists(short_tmpdir)


def test_run_one_cleans_short_tmpdir_when_entrypoint_cannot_start(
    monkeypatch, audit_test_root: Path
) -> None:
    token = f"fail{os.getpid()}"
    expected_link = Path("/tmp") / f"dlh-{os.getpid()}-{token}"
    monkeypatch.setattr(full_run.secrets, "token_hex", lambda _: token)

    def fail_to_start(*args: Any, **kwargs: Any) -> None:
        raise OSError("synthetic start failure")

    monkeypatch.setattr(
        full_run,
        "build_full_command",
        lambda *args, **kwargs: [sys.executable, "-c", "pass"],
    )
    monkeypatch.setattr(full_run.subprocess, "Popen", fail_to_start)
    bootstrap = full_run._prepare_bootstrap(audit_test_root, allow_network=False)
    result = full_run._run_one(
        {"lesson_id": "demo/fail", "track": "demo", "lesson": "fail", "kind": "run"},
        root=full_run.repo_root(),
        run_root=audit_test_root,
        run_name="full-fail-a001",
        attempt=1,
        device="cpu",
        cuda_index=0,
        cpu_threads=1,
        timeout_seconds=1,
        bootstrap=bootstrap,
        allow_network=False,
    )

    assert result["state"] == "failed_start"
    assert "synthetic start failure" in result["error"]
    assert not os.path.lexists(expected_link)


def test_cuda_lock_excludes_a_second_audit(audit_test_root: Path) -> None:
    with full_run._cuda_execution_lock(
        audit_test_root,
        enabled=True,
        timeout_seconds=1,
        run_id="outer",
    ) as lock_path:
        assert lock_path == audit_test_root / "cuda.lock"
        assert json.loads(lock_path.read_text(encoding="utf-8"))["run_id"] == "outer"
        with pytest.raises(TimeoutError, match="waiting for"):
            with full_run._cuda_execution_lock(
                audit_test_root,
                enabled=True,
                timeout_seconds=0.01,
                run_id="inner",
            ):
                pytest.fail("second CUDA audit acquired a held lock")


def test_failed_retry_uses_a_new_attempt_directory(monkeypatch, audit_test_root: Path) -> None:
    spec = {"lesson_id": "demo/lesson", "estimated_train_batches": 1}
    manifest = {
        "inventory_sha256": "inventory",
        "source_snapshot": {"source_sha256": "source"},
        "lessons": [spec],
    }
    report_path = audit_test_root / "runs" / "retry" / "report.json"
    run_names: list[tuple[str, int]] = []

    monkeypatch.setattr(full_run, "_assert_source_snapshot", lambda root, expected: None)

    def fake_run_one(spec: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        run_names.append((kwargs["run_name"], kwargs["attempt"]))
        state = "failed_exit" if len(run_names) == 1 else "passed"
        return {
            "state": state,
            "attempt": kwargs["attempt"],
            "run_name": kwargs["run_name"],
            "elapsed_seconds": 0.01,
            "returncode": 1 if state == "failed_exit" else 0,
            "error": "failure" if state == "failed_exit" else None,
            "interrupted": False,
        }

    monkeypatch.setattr(full_run, "_run_one", fake_run_one)
    common = {
        "manifest": manifest,
        "selected": [spec],
        "root": full_run.repo_root(),
        "runtime_root": audit_test_root,
        "report_path": report_path,
        "run_id": "retry",
        "device": "cpu",
        "device_details": {"resolved": "cpu"},
        "cuda_index": 0,
        "cpu_threads": 1,
        "timeout_seconds": 1,
        "cuda_lock_timeout_seconds": 1,
        "allow_network": False,
    }
    first = full_run.run_manifest_selection(
        **common,
        resume=False,
        retry_failed=False,
    )
    assert first["status"] == "finished_with_failures"

    second = full_run.run_manifest_selection(
        **common,
        resume=True,
        retry_failed=True,
    )
    entry = second["lessons"]["demo/lesson"]
    assert second["status"] == "complete"
    expected_names = [
        full_run._attempt_run_name("full-retry", "demo/lesson", attempt) for attempt in (1, 2)
    ]
    assert run_names == list(zip(expected_names, (1, 2)))
    assert entry["attempts"] == 2
    assert [attempt["run_name"] for attempt in entry["attempt_history"]] == expected_names


def test_source_drift_stops_before_starting_next_lesson(monkeypatch, audit_test_root: Path) -> None:
    spec = {"lesson_id": "demo/lesson", "estimated_train_batches": 1}
    manifest = {
        "inventory_sha256": "inventory",
        "source_snapshot": {"source_sha256": "source"},
        "lessons": [spec],
    }
    checks = 0

    def assert_snapshot(root: Path, expected: dict[str, Any]) -> None:
        nonlocal checks
        checks += 1
        if checks == 2:
            raise RuntimeError("lesson runtime source changed: expected source, got changed")

    monkeypatch.setattr(full_run, "_assert_source_snapshot", assert_snapshot)
    monkeypatch.setattr(
        full_run,
        "_run_one",
        lambda *args, **kwargs: pytest.fail("source-drifted lesson must not start"),
    )
    report = full_run.run_manifest_selection(
        manifest,
        [spec],
        root=full_run.repo_root(),
        runtime_root=audit_test_root,
        report_path=audit_test_root / "runs" / "drift" / "report.json",
        run_id="drift",
        device="cpu",
        device_details={"resolved": "cpu"},
        cuda_index=0,
        cpu_threads=1,
        timeout_seconds=1,
        cuda_lock_timeout_seconds=1,
        allow_network=False,
        resume=False,
        retry_failed=False,
    )

    assert report["status"] == "source_changed"
    assert report["summary"] == {"pending": 1, "total": 1}
    assert "expected source" in report["source_change_error"]
