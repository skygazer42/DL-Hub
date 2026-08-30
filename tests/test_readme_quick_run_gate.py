"""Runtime gate for the 35 lesson README quick-run commands added in 2026-08.

The expensive parameterized tests are opt-in so the normal unit-test suite does
not train 35 models. Run the complete gate from the repository root with:

    DLHUB_RUN_README_GATE=1 pytest -q -s tests/test_readme_quick_run_gate.py

Each case executes the command documented in that lesson's README, changing
only ``--max-train-batches`` and ``--max-eval-batches`` to one. Outputs live in
one session-scoped temporary directory and are removed when the session ends.
"""

from __future__ import annotations

import json
import math
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import torch


@dataclass(frozen=True)
class ReadmeQuickRun:
    track: str
    lesson: str

    @property
    def id(self) -> str:
        return f"{self.track}/{self.lesson}"

    @property
    def module(self) -> str:
        return f"tracks.{self.track}.{self.lesson}.train"


README_QUICK_RUNS = (
    ReadmeQuickRun("generative", "lesson_09_compact_conditional_gan"),
    ReadmeQuickRun("generative", "lesson_12_compact_layout_to_image"),
    ReadmeQuickRun("generative", "lesson_13_compact_text_to_image_diffusion"),
    ReadmeQuickRun("generative", "lesson_14_compact_diffusion_inpainting"),
    ReadmeQuickRun("generative", "lesson_46_compact_image_to_video_diffusion"),
    ReadmeQuickRun("llm", "lesson_09_compact_rlhf_ppo"),
    ReadmeQuickRun("multimodal", "lesson_24_multimodal_reasoning"),
    ReadmeQuickRun("multimodal", "lesson_50_person_pose_vlm_reasoning"),
    ReadmeQuickRun("nlp", "lesson_15_compact_cross_encoder_reranking"),
    ReadmeQuickRun("nlp", "lesson_16_compact_text_clustering"),
    ReadmeQuickRun("nlp", "lesson_17_compact_text_anomaly_detection"),
    ReadmeQuickRun("nlp", "lesson_18_compact_topic_modeling"),
    ReadmeQuickRun("nlp", "lesson_19_compact_distilled_text_classifier"),
    ReadmeQuickRun("nlp", "lesson_20_compact_adversarial_text_classification"),
    ReadmeQuickRun("nlp", "lesson_21_compact_adversarial_example_detection"),
    ReadmeQuickRun("vision", "lesson_25_synthetic_image_fusion"),
    ReadmeQuickRun("vision", "lesson_26_synthetic_text_detection"),
    ReadmeQuickRun("vision", "lesson_27_synthetic_edge_detection"),
    ReadmeQuickRun("vision", "lesson_28_synthetic_salient_object_detection"),
    ReadmeQuickRun("vision", "lesson_29_synthetic_camouflaged_object_detection"),
    ReadmeQuickRun("vision", "lesson_30_synthetic_salient_object_detection_boxes"),
    ReadmeQuickRun("vision", "lesson_31_synthetic_interactive_segmentation"),
    ReadmeQuickRun("vision", "lesson_60_synthetic_image_deraining"),
    ReadmeQuickRun("vision", "lesson_66_synthetic_video_object_detection"),
    ReadmeQuickRun("vision", "lesson_67_synthetic_video_stabilization"),
    ReadmeQuickRun("vision", "lesson_68_synthetic_video_frame_interpolation"),
    ReadmeQuickRun("vision", "lesson_69_synthetic_video_restoration"),
    ReadmeQuickRun("vision", "lesson_70_synthetic_video_understanding"),
    ReadmeQuickRun("vision", "lesson_71_synthetic_video_summarization"),
    ReadmeQuickRun("vision", "lesson_72_synthetic_video_enhancement"),
    ReadmeQuickRun("vision", "lesson_73_synthetic_video_object_segmentation"),
    ReadmeQuickRun("vision", "lesson_74_synthetic_video_instance_segmentation"),
    ReadmeQuickRun("vision", "lesson_75_synthetic_video_matting"),
    ReadmeQuickRun("vision", "lesson_86_synthetic_co_segmentation"),
    ReadmeQuickRun("vision", "lesson_87_synthetic_action_recognition"),
)

COMMON_ARTIFACTS = (
    "config.json",
    "metrics.jsonl",
    "logs/train.log",
    "checkpoints/checkpoint.pt",
)

JSON_ARTIFACTS = {
    ("llm", "lesson_09_compact_rlhf_ppo"): ("vocab.json",),
    ("multimodal", "lesson_50_person_pose_vlm_reasoning"): ("vocab.json",),
    ("nlp", "lesson_15_compact_cross_encoder_reranking"): ("vocab.json",),
    ("nlp", "lesson_16_compact_text_clustering"): ("vocab.json",),
    ("nlp", "lesson_17_compact_text_anomaly_detection"): ("vocab.json",),
    ("nlp", "lesson_18_compact_topic_modeling"): ("vocab.json",),
    ("nlp", "lesson_19_compact_distilled_text_classifier"): ("vocab.json",),
    ("nlp", "lesson_20_compact_adversarial_text_classification"): ("vocab.json",),
    ("nlp", "lesson_21_compact_adversarial_example_detection"): ("vocab.json",),
}

JSONL_ARTIFACTS = {
    ("llm", "lesson_09_compact_rlhf_ppo"): ("samples.jsonl",),
}

# Generative artifacts are checked for loadability, expected task-specific
# fields, non-empty tensors, and finite floating-point values.
TORCH_ARTIFACTS = {
    ("generative", "lesson_09_compact_conditional_gan"): {
        "samples.pt": ("samples", "labels"),
    },
    ("generative", "lesson_12_compact_layout_to_image"): {
        "samples.pt": ("layout", "target", "prediction"),
    },
    ("generative", "lesson_13_compact_text_to_image_diffusion"): {
        "samples.pt": ("token_ids", "samples"),
        "denoise_grid.pt": ("frames",),
    },
    ("generative", "lesson_14_compact_diffusion_inpainting"): {
        "samples.pt": ("context", "target", "mask", "samples"),
        "denoise_grid.pt": ("frames",),
    },
    ("generative", "lesson_46_compact_image_to_video_diffusion"): {
        "samples.pt": ("source", "target_video", "samples"),
        "trajectory.pt": ("trajectory",),
    },
}

_RUNTIME_RESULTS: list[tuple[str, float]] = []


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _read_documented_command(case: ReadmeQuickRun) -> list[str]:
    readme = _repo_root() / "tracks" / case.track / case.lesson / "README.md"
    lines = readme.read_text(encoding="utf-8").splitlines()
    commands: list[list[str]] = []
    index = 0
    while index < len(lines):
        current = lines[index].strip()
        if not current.startswith(("python -m tracks.", "python3 -m tracks.")):
            index += 1
            continue
        parts: list[str] = []
        while True:
            current = lines[index].strip()
            continued = current.endswith("\\")
            parts.append(current[:-1].strip() if continued else current)
            index += 1
            if not continued or index >= len(lines):
                break
        commands.append(shlex.split(" ".join(parts)))

    assert len(commands) == 1, f"{case.id}: expected one documented python -m command"
    tokens = commands[0]
    assert tokens[1:3] == ["-m", case.module], (
        f"{case.id}: README command targets {tokens[1:3]!r}, expected {case.module}"
    )
    return tokens


def _flag_value(tokens: list[str], flag: str) -> str | None:
    for index, token in enumerate(tokens):
        if token == flag:
            assert index + 1 < len(tokens), f"{flag} is missing its value"
            return tokens[index + 1]
        if token.startswith(f"{flag}="):
            return token.split("=", 1)[1]
    return None


def _runtime_command(case: ReadmeQuickRun) -> list[str]:
    tokens = _read_documented_command(case)
    assert _flag_value(tokens, "--device") == "cpu"
    assert _flag_value(tokens, "--epochs") == "1"
    assert _flag_value(tokens, "--run-name") == "smoke"

    tokens[0] = sys.executable
    for flag in ("--max-train-batches", "--max-eval-batches"):
        assert _flag_value(tokens, flag) is not None, (
            f"{case.id}: README command must include {flag} so the runtime gate stays bounded"
        )
        replaced = False
        for index, token in enumerate(tokens):
            if token == flag:
                tokens[index + 1] = "1"
                replaced = True
            elif token.startswith(f"{flag}="):
                tokens[index] = f"{flag}=1"
                replaced = True
        assert replaced and _flag_value(tokens, flag) == "1", (
            f"{case.id}: runtime gate could not force {flag}=1"
        )
    return tokens


def _iter_numbers(value: Any) -> Iterator[float]:
    if isinstance(value, bool):
        return
    if isinstance(value, int | float):
        yield float(value)
    elif isinstance(value, dict):
        for child in value.values():
            yield from _iter_numbers(child)
    elif isinstance(value, list):
        for child in value:
            yield from _iter_numbers(child)


def _assert_jsonl_finite(path: Path) -> int:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert lines, f"{path}: expected at least one JSONL record"
    numeric_values: list[float] = []
    for line_number, line in enumerate(lines, start=1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            pytest.fail(f"{path}:{line_number}: invalid JSON: {exc}")
        numeric_values.extend(_iter_numbers(record))
    assert numeric_values, f"{path}: expected numeric metrics"
    assert all(math.isfinite(value) for value in numeric_values), f"{path}: non-finite metric"
    return len(lines)


def _assert_metric_keys_finite(path: Path, required_keys: tuple[str, ...]) -> None:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert lines, f"{path}: expected at least one JSONL record"
    for line_number, line in enumerate(lines, start=1):
        record = json.loads(line)
        assert isinstance(record, Mapping), f"{path}:{line_number}: expected a JSON object"
        missing = set(required_keys).difference(record)
        assert not missing, f"{path}:{line_number}: missing metrics {sorted(missing)}"
        for key in required_keys:
            value = record[key]
            assert not isinstance(value, bool) and isinstance(value, int | float), (
                f"{path}:{line_number}: {key} must be numeric"
            )
            assert math.isfinite(float(value)), f"{path}:{line_number}: non-finite {key}"


def _assert_tensors_finite(value: Any, label: str) -> int:
    tensor_count = 0
    if isinstance(value, torch.Tensor):
        assert value.numel() > 0, f"{label}: empty tensor"
        if value.is_floating_point() or value.is_complex():
            assert torch.isfinite(value).all().item(), f"{label}: non-finite tensor"
        return 1
    if isinstance(value, dict):
        for key, child in value.items():
            tensor_count += _assert_tensors_finite(child, f"{label}.{key}")
    elif isinstance(value, list | tuple):
        for index, child in enumerate(value):
            tensor_count += _assert_tensors_finite(child, f"{label}[{index}]")
    return tensor_count


def _safe_torch_load(path: Path) -> Any:
    return torch.load(path, map_location="cpu", weights_only=True)


def _assert_checkpoint(case: ReadmeQuickRun, path: Path) -> None:
    payload = _safe_torch_load(path)
    assert isinstance(payload, Mapping), f"{path}: expected a checkpoint mapping"
    required_keys = {"model_state", "epoch", "extra"}
    assert required_keys.issubset(payload), (
        f"{path}: missing keys {sorted(required_keys.difference(payload))}"
    )
    assert payload["epoch"] == 1, f"{path}: expected epoch 1"
    assert isinstance(payload["model_state"], Mapping) and payload["model_state"], (
        f"{path}: model_state must be a non-empty mapping"
    )
    assert _assert_tensors_finite(payload["model_state"], "checkpoint.model_state") > 0
    assert isinstance(payload["extra"], Mapping), f"{path}: extra must be a mapping"
    assert payload["extra"].get("track") == case.track
    assert payload["extra"].get("lesson") == case.lesson


def _assert_special_artifacts(case: ReadmeQuickRun, run_dir: Path) -> None:
    key = (case.track, case.lesson)
    for relative in JSON_ARTIFACTS.get(key, ()):
        artifact = run_dir / relative
        assert artifact.is_file() and artifact.stat().st_size > 0, f"missing {artifact}"
        json.loads(artifact.read_text(encoding="utf-8"))

    for relative in JSONL_ARTIFACTS.get(key, ()):
        artifact = run_dir / relative
        assert artifact.is_file() and artifact.stat().st_size > 0, f"missing {artifact}"
        _assert_jsonl_finite(artifact)

    for relative, expected_keys in TORCH_ARTIFACTS.get(key, {}).items():
        artifact = run_dir / relative
        assert artifact.is_file() and artifact.stat().st_size > 0, f"missing {artifact}"
        payload = _safe_torch_load(artifact)
        assert isinstance(payload, dict), f"{artifact}: expected a dictionary"
        assert set(expected_keys).issubset(payload), (
            f"{artifact}: missing keys {sorted(set(expected_keys).difference(payload))}"
        )
        assert _assert_tensors_finite(payload, relative) >= len(expected_keys)

    if key == ("generative", "lesson_09_compact_conditional_gan"):
        _assert_metric_keys_finite(
            run_dir / "metrics.jsonl",
            ("d_loss", "g_loss", "val_d_loss", "val_g_loss"),
        )
        payload = _safe_torch_load(run_dir / "samples.pt")
        samples, labels = payload["samples"], payload["labels"]
        assert samples.ndim == 4 and samples.shape[0] == labels.shape[0]
        assert float(samples.min()) >= -1.001 and float(samples.max()) <= 1.001
    elif key == ("generative", "lesson_12_compact_layout_to_image"):
        payload = _safe_torch_load(run_dir / "samples.pt")
        layout, target, prediction = payload["layout"], payload["target"], payload["prediction"]
        assert layout.shape[0] == target.shape[0] == prediction.shape[0]
        assert target.shape == prediction.shape
    elif key == ("generative", "lesson_46_compact_image_to_video_diffusion"):
        payload = _safe_torch_load(run_dir / "samples.pt")
        assert payload["target_video"].shape == payload["samples"].shape
        assert payload["source"].shape[0] == payload["samples"].shape[0]


@pytest.fixture(scope="session")
def readme_gate_output_root() -> Iterator[Path]:
    path = Path(tempfile.mkdtemp(prefix="dlhub-readme-runtime-gate-"))
    print(f"README_RUNTIME_GATE outputs={path}")
    try:
        yield path
    finally:
        total_seconds = sum(elapsed for _, elapsed in _RUNTIME_RESULTS)
        print(
            f"README_RUNTIME_GATE summary={len(_RUNTIME_RESULTS)}/{len(README_QUICK_RUNS)} "
            f"subprocess_seconds={total_seconds:.2f}"
        )
        shutil.rmtree(path)
        print(f"README_RUNTIME_GATE cleaned={path}")


def test_readme_runtime_gate_inventory() -> None:
    assert len(README_QUICK_RUNS) == 35
    assert len({case.id for case in README_QUICK_RUNS}) == len(README_QUICK_RUNS)
    for case in README_QUICK_RUNS:
        _runtime_command(case)


def test_conditional_gan_metric_acceptance(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.jsonl"
    required = ("d_loss", "g_loss", "val_d_loss", "val_g_loss")
    record = {key: 0.5 for key in required}
    metrics_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    _assert_metric_keys_finite(metrics_path, required)

    metrics_path.write_text(
        json.dumps({key: 0.5 for key in required[:-1]}) + "\n", encoding="utf-8"
    )
    with pytest.raises(AssertionError, match="missing metrics.*val_g_loss"):
        _assert_metric_keys_finite(metrics_path, required)

    record["val_g_loss"] = math.nan
    metrics_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(AssertionError, match="non-finite val_g_loss"):
        _assert_metric_keys_finite(metrics_path, required)


@pytest.mark.skipif(
    os.environ.get("DLHUB_RUN_README_GATE") != "1",
    reason="set DLHUB_RUN_README_GATE=1 to execute all 35 documented quick runs",
)
@pytest.mark.parametrize("case", README_QUICK_RUNS, ids=lambda case: case.id)
def test_documented_readme_quick_run(
    case: ReadmeQuickRun,
    readme_gate_output_root: Path,
) -> None:
    command = _runtime_command(case)
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "DLHUB_OUTPUTS_DIR": str(readme_gate_output_root),
            "HF_HUB_OFFLINE": "1",
            "MKL_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "PYTHONHASHSEED": "0",
            "TRANSFORMERS_OFFLINE": "1",
            "WANDB_MODE": "offline",
        }
    )

    started = time.monotonic()
    proc = subprocess.run(
        command,
        cwd=_repo_root(),
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
    )
    elapsed = time.monotonic() - started
    assert proc.returncode == 0, (
        f"{case.id}: exit={proc.returncode}\n"
        f"command: {shlex.join(command)}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )

    run_dir = readme_gate_output_root / case.track / case.lesson / "smoke"
    for relative in COMMON_ARTIFACTS:
        artifact = run_dir / relative
        assert artifact.is_file(), f"{case.id}: missing {artifact}"
        assert artifact.stat().st_size > 0, f"{case.id}: empty {artifact}"
    _assert_checkpoint(case, run_dir / "checkpoints/checkpoint.pt")

    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    assert "cpu" in {
        value for value in _walk_key_values(config, "device") if isinstance(value, str)
    }, f"{case.id}: config does not record CPU execution"
    metric_records = _assert_jsonl_finite(run_dir / "metrics.jsonl")
    _assert_special_artifacts(case, run_dir)

    _RUNTIME_RESULTS.append((case.id, elapsed))
    print(
        f"README_RUNTIME_GATE pass={case.id} exit=0 metrics={metric_records} elapsed={elapsed:.2f}s"
    )


def _walk_key_values(value: Any, key: str) -> Iterator[Any]:
    if isinstance(value, dict):
        for child_key, child in value.items():
            if child_key == key:
                yield child
            yield from _walk_key_values(child, key)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_key_values(child, key)
