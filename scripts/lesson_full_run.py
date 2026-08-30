"""Inventory and orchestrate complete default-budget runs for all lessons.

"Complete" means the lesson's normal offline entrypoint with its declared
default epochs, steps, batch size, and data scale.  The runner injects only the
execution device, an isolated run name, and ``--dataset fake`` for the seven
lessons whose offline route is explicit.  It never injects ``--max-*`` or any
training-budget override.

Runtime state lives under ``outputs/runtime-audit`` on the repository's
``/data`` filesystem.  Every lesson receives an isolated cwd, temp directory,
cache tree, output tree, stdout/stderr logs, timeout, and atomically persisted
terminal state so an interrupted audit can be resumed.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import fcntl
import hashlib
import json
import math
import os
import re
import secrets
import signal
import subprocess
import sys
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from scripts.lesson_contracts import (
        OFFLINE_EXPLICIT_FAKE,
        LessonContract,
        discover_lesson_contracts,
    )
except ModuleNotFoundError:  # Direct ``python scripts/...`` execution.
    from lesson_contracts import (
        OFFLINE_EXPLICIT_FAKE,
        LessonContract,
        discover_lesson_contracts,
    )

SCHEMA_VERSION = 1
ARTIFACT_PATTERN = re.compile(
    r"^(?:[A-Za-z0-9_.-]+/)*[A-Za-z0-9_.-]+\."
    r"(?:jsonl|json|pt|pth|png|jpg|jpeg|csv|txt|log|npy|npz|wav|mp4)$",
    re.IGNORECASE,
)
DOCUMENTED_ARTIFACT_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_.-])(?:[A-Za-z0-9_.-]+/)*[A-Za-z0-9_.-]+\."
    r"(?:jsonl|json|pt|pth|png|jpg|jpeg|csv|txt|log|npy|npz|wav|mp4)\b",
    re.IGNORECASE,
)
BENCHMARK_PATTERN = re.compile(
    r"benchmark|基准|state[- ]of[- ]the[- ]art|\bSOTA\b|论文.{0,12}复现|"
    r"paper[- ]faithful|(?:accuracy|准确率|\bF1\b|mIoU).{0,20}\d",
    re.IGNORECASE,
)
BENCHMARK_DISCLAIMER_PATTERN = re.compile(
    r"not.{0,12}benchmark|不(?:是|代表|等于).{0,12}(?:benchmark|基准|论文复现)|"
    r"不自动代表|does not (?:prove|imply)",
    re.IGNORECASE,
)
ACCEPTANCE_RANGE_PATTERN = re.compile(
    r"(?:accuracy|准确率|IoU).{0,80}\[\s*0(?:\.0+)?\s*,\s*1(?:\.0+)?\s*\]",
    re.IGNORECASE,
)
LOCAL_OFFLINE_BENCHMARK_PATTERN = re.compile(
    r"benchmark.{0,40}(?:without downloaded|synthetic|offline)", re.IGNORECASE
)
NETWORK_PATTERNS = (
    ("requests", re.compile(r"\brequests\s*\.")),
    ("urllib", re.compile(r"\burllib(?:\.request)?\s*\.")),
    ("http-client", re.compile(r"\bhttpx?\s*\.")),
    ("socket", re.compile(r"\bsocket\s*\.")),
    ("url", re.compile(r"https?://")),
    ("download-enabled", re.compile(r"\bdownload\s*=\s*True\b")),
    ("pretrained-fetch", re.compile(r"\bfrom_pretrained\s*\(")),
    ("external-api", re.compile(r"\b(?:openai|boto3|wandb)\s*\.")),
)
TRAINING_BUDGET_FLAGS = {
    "--epochs",
    "--steps",
    "--steps-per-epoch",
    "--num-episodes",
}
ALGORITHM_STEP_FLAGS = {
    "--num-audio-steps",
    "--num-diffusion-steps",
    "--num-discretization-steps",
    "--num-sample-steps",
    "--sample-steps",
}
DATA_SCALE_FLAGS = {
    "--batch-size",
    "--image-size",
    "--max-doc-length",
    "--max-length",
    "--max-query-length",
    "--max-text-length",
    "--num-docs",
    "--num-frames",
    "--num-graphs",
    "--num-items",
    "--num-nodes",
    "--num-papers",
    "--num-points",
    "--num-prompts",
    "--num-samples",
    "--num-users",
    "--seq-length",
    "--sequence-length",
    "--text-length",
    "--val-fraction",
}
ALLOWED_INJECTED_FLAGS = {"--dataset", "--device", "--run-name"}
TRAINING_LIMIT_FLAGS = {"--max-eval-batches", "--max-train-batches"}
TERMINAL_FAILURE_STATES = {"failed_exit", "failed_start", "failed_validation", "timed_out"}
SOURCE_PATHS = (
    "dlhub",
    "tracks",
    "ml_algorithms",
    "optimization",
    "scripts/lesson_contracts.py",
    "scripts/lesson_full_run.py",
    "pyproject.toml",
)
SOURCE_SUFFIXES = {".md", ".py", ".toml"}
RUNTIME_SITE_CUSTOMIZE = """\
import atexit
import json
import os
import sys

def _record_torch_peak_memory():
    torch = sys.modules.get("torch")
    metrics_dir = os.environ.get("DLHUB_RUNTIME_AUDIT_GPU_METRICS_DIR")
    if torch is None or not metrics_dir:
        return
    try:
        if not torch.cuda.is_available():
            return
        os.makedirs(metrics_dir, exist_ok=True)
        payload = {
            "pid": os.getpid(),
            "metric": "torch.cuda.max_memory_allocated",
            "devices": {
                str(index): int(torch.cuda.max_memory_allocated(index))
                for index in range(torch.cuda.device_count())
            },
        }
        path = os.path.join(metrics_dir, f"{os.getpid()}.json")
        temporary = f"{path}.tmp"
        with open(temporary, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, sort_keys=True)
        os.replace(temporary, path)
    except Exception:
        pass

atexit.register(_record_torch_peak_memory)
"""
OFFLINE_SOCKET_BLOCK = """\
import socket

_original_connect = socket.socket.connect
_original_connect_ex = socket.socket.connect_ex

def _blocked_connect(self, address):
    if self.family in (socket.AF_INET, socket.AF_INET6):
        raise RuntimeError("DL-Hub runtime audit blocked AF_INET/AF_INET6 socket access")
    return _original_connect(self, address)

def _blocked_connect_ex(self, address):
    if self.family in (socket.AF_INET, socket.AF_INET6):
        raise RuntimeError("DL-Hub runtime audit blocked AF_INET/AF_INET6 socket access")
    return _original_connect_ex(self, address)

def _blocked_create_connection(*args, **kwargs):
    raise RuntimeError("DL-Hub runtime audit blocked socket.create_connection")

socket.socket.connect = _blocked_connect
socket.socket.connect_ex = _blocked_connect_ex
socket.create_connection = _blocked_create_connection
"""


@dataclass(frozen=True)
class LessonRunSpec:
    track: str
    lesson: str
    entrypoint: str
    module: str
    kind: str
    offline_mode: str
    cli_defaults: dict[str, Any]
    cli_choices: dict[str, list[Any]]
    training_budget_defaults: dict[str, Any]
    algorithm_step_defaults: dict[str, Any]
    data_scale_defaults: dict[str, Any]
    limit_defaults: dict[str, Any]
    command_template: tuple[str, ...]
    estimated_train_batches: int | None
    estimate_basis: str | None
    resource_band: str
    required_artifacts: tuple[str, ...]
    artifact_candidates: tuple[str, ...]
    documented_artifacts: tuple[str, ...]
    third_party_imports: tuple[str, ...]
    network_indicators: tuple[str, ...]
    external_dependency_class: str
    benchmark_classification: str
    benchmark_mentions: tuple[str, ...]

    @property
    def lesson_id(self) -> str:
        return f"{self.track}/{self.lesson}"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_runtime_root() -> Path:
    return repo_root() / "outputs" / "runtime-audit"


def _source_files(root: Path) -> list[Path]:
    files: set[Path] = set()
    for relative in SOURCE_PATHS:
        target = root / relative
        if target.is_file():
            files.add(target)
            continue
        if target.is_dir():
            files.update(
                path
                for path in target.rglob("*")
                if path.is_file()
                and path.suffix in SOURCE_SUFFIXES
                and "__pycache__" not in path.parts
            )
    return sorted(files, key=lambda path: path.relative_to(root).as_posix())


def _source_tree_digest(root: Path) -> tuple[str, int, int]:
    digest = hashlib.sha256()
    byte_count = 0
    files = _source_files(root)
    for path in files:
        relative = path.relative_to(root).as_posix().encode()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        file_digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                byte_count += len(chunk)
                file_digest.update(chunk)
        digest.update(file_digest.digest())
    return digest.hexdigest(), len(files), byte_count


def _git_output(root: Path, *args: str) -> bytes:
    result = subprocess.run(
        ["git", *args],
        cwd=root,
        check=False,
        capture_output=True,
    )
    return result.stdout if result.returncode == 0 else b""


def _source_snapshot(root: Path) -> dict[str, Any]:
    source_sha256, file_count, byte_count = _source_tree_digest(root)
    git_head = _git_output(root, "rev-parse", "HEAD").decode(errors="replace").strip() or None
    diff = _git_output(root, "diff", "--binary", "HEAD", "--", *SOURCE_PATHS)
    status = _git_output(
        root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--",
        *SOURCE_PATHS,
    )
    return {
        "source_sha256": source_sha256,
        "source_file_count": file_count,
        "source_bytes": byte_count,
        "git_head": git_head,
        "git_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "git_status_sha256": hashlib.sha256(status).hexdigest(),
        "git_status": status.decode(encoding="utf-8", errors="replace").splitlines(),
        "included_paths": list(SOURCE_PATHS),
    }


def _assert_source_snapshot(root: Path, expected: dict[str, Any]) -> None:
    actual_sha256, actual_count, actual_bytes = _source_tree_digest(root)
    if actual_sha256 != expected["source_sha256"]:
        raise RuntimeError(
            "lesson runtime source changed: "
            f"expected {expected['source_sha256']}, got {actual_sha256} "
            f"({actual_count} files / {actual_bytes} bytes)"
        )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _literal(node: ast.AST | None) -> Any:
    if node is None:
        return None
    try:
        return ast.literal_eval(node)
    except (TypeError, ValueError):
        return {"dynamic": ast.unparse(node)}


def _call_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    if isinstance(node.func, ast.Name):
        return node.func.id
    return None


def _inspect_cli(path: Path) -> tuple[dict[str, Any], dict[str, list[Any]]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    defaults: dict[str, Any] = {}
    choices: dict[str, list[Any]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or _call_name(node) != "add_argument":
            continue
        flags = [
            arg.value
            for arg in node.args
            if isinstance(arg, ast.Constant)
            and isinstance(arg.value, str)
            and arg.value.startswith("--")
        ]
        if not flags:
            continue
        flag = flags[0]
        keywords = {keyword.arg: keyword.value for keyword in node.keywords if keyword.arg}
        action = _literal(keywords.get("action"))
        if "default" in keywords:
            default = _literal(keywords["default"])
        elif action == "store_true":
            default = False
        elif action == "store_false":
            default = True
        else:
            default = None
        defaults[flag] = default
        raw_choices = _literal(keywords.get("choices"))
        if isinstance(raw_choices, list | tuple):
            choices[flag] = list(raw_choices)
    return dict(sorted(defaults.items())), dict(sorted(choices.items()))


def _source_artifacts(path: Path) -> tuple[str, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return tuple(
        sorted(
            {
                node.value
                for node in ast.walk(tree)
                if isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and ARTIFACT_PATTERN.fullmatch(node.value)
            }
        )
    )


def _required_artifacts(kind: str) -> tuple[str, ...]:
    if kind == "run":
        return ()
    return (
        "config.json",
        "metrics.jsonl",
        "logs/train.log",
        "checkpoints/checkpoint.pt",
    )


def _documented_artifacts(readme: Path) -> tuple[str, ...]:
    if not readme.is_file():
        return ()
    return tuple(
        sorted(set(DOCUMENTED_ARTIFACT_PATTERN.findall(readme.read_text(encoding="utf-8"))))
    )


def _benchmark_audit(readme: Path) -> tuple[str, tuple[str, ...]]:
    if not readme.is_file():
        return "undocumented", ()
    mentions = tuple(
        f"{line_number}: {line.strip()}"
        for line_number, line in enumerate(readme.read_text(encoding="utf-8").splitlines(), 1)
        if BENCHMARK_PATTERN.search(line)
    )
    if not mentions:
        return "none", ()
    if all(BENCHMARK_DISCLAIMER_PATTERN.search(mention) for mention in mentions):
        return "disclaimer-only", mentions
    if all(ACCEPTANCE_RANGE_PATTERN.search(mention) for mention in mentions):
        return "acceptance-range-only", mentions
    if all(LOCAL_OFFLINE_BENCHMARK_PATTERN.search(mention) for mention in mentions):
        return "local-offline-benchmark", mentions
    return "review-required", mentions


def _source_dependency_audit(lesson_dir: Path) -> tuple[tuple[str, ...], tuple[str, ...]]:
    imports: set[str] = set()
    indicators: list[str] = []
    local_roots = {"dlhub", "tracks", "ml_algorithms", "optimization"}
    stdlib = getattr(sys, "stdlib_module_names", set())
    for path in sorted(lesson_dir.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(text, filename=str(path))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module.split(".", 1)[0])
        for line_number, line in enumerate(text.splitlines(), 1):
            for label, pattern in NETWORK_PATTERNS:
                if pattern.search(line):
                    indicators.append(f"{path.name}:{line_number}:{label}: {line.strip()[:180]}")
    third_party = tuple(sorted(imports - local_roots - set(stdlib) - {"__future__"}))
    return third_party, tuple(indicators)


def _estimate_batches(defaults: dict[str, Any]) -> tuple[int | None, str | None, str]:
    samples = defaults.get("--num-samples")
    batch_size = defaults.get("--batch-size")
    epochs = defaults.get("--epochs")
    if not isinstance(samples, int) or not isinstance(batch_size, int) or batch_size <= 0:
        return None, None, "unknown"
    if not isinstance(epochs, int) or epochs <= 0:
        return None, None, "unknown"
    batches = math.ceil(samples / batch_size) * epochs
    if batches > 2_000:
        band = "extreme"
    elif batches > 500:
        band = "high"
    elif batches > 100:
        band = "medium"
    else:
        band = "low"
    return batches, "ceil(num_samples / batch_size) * epochs", band


def _command_template(contract: LessonContract, defaults: dict[str, Any]) -> tuple[str, ...]:
    if contract.entrypoint_module is None:
        raise ValueError(f"{contract.key}: missing entrypoint module")
    command = ["{python}", "-m", contract.entrypoint_module]
    if "--device" in defaults:
        command.extend(("--device", "{device}"))
    if "--run-name" in defaults:
        command.extend(("--run-name", "{run_name}"))
    if contract.offline_mode == OFFLINE_EXPLICIT_FAKE:
        command.extend(("--dataset", "fake"))
    injected = {token for token in command if token.startswith("--")}
    forbidden = injected - ALLOWED_INJECTED_FLAGS
    if forbidden or any(flag.startswith("--max-") for flag in injected):
        raise ValueError(
            f"{contract.key}: full command has forbidden overrides {sorted(forbidden)}"
        )
    return tuple(command)


def build_lesson_manifest(root: Path | None = None) -> dict[str, Any]:
    root = (root or repo_root()).resolve()
    source_snapshot = _source_snapshot(root)
    specs: list[LessonRunSpec] = []
    for contract in discover_lesson_contracts(root):
        if contract.entrypoint is None or contract.entrypoint_module is None:
            raise ValueError(f"{contract.key}: missing entrypoint")
        entrypoint = root / contract.entrypoint
        lesson_dir = entrypoint.parent
        defaults, choices = _inspect_cli(entrypoint)
        artifacts = _source_artifacts(entrypoint)
        third_party, indicators = _source_dependency_audit(lesson_dir)
        benchmark_classification, benchmark_mentions = _benchmark_audit(lesson_dir / "README.md")
        estimated_batches, estimate_basis, resource_band = _estimate_batches(defaults)
        if contract.offline_mode == OFFLINE_EXPLICIT_FAKE:
            external_class = "optional-real-data; orchestrated command selects fake"
        elif indicators:
            external_class = "network-sensitive source detected; orchestrated sockets blocked"
        else:
            external_class = "no external data/service requirement detected"
        spec = LessonRunSpec(
            track=contract.track,
            lesson=contract.lesson,
            entrypoint=contract.entrypoint,
            module=contract.entrypoint_module,
            kind=contract.entrypoint_kind or "unknown",
            offline_mode=contract.offline_mode,
            cli_defaults=defaults,
            cli_choices=choices,
            training_budget_defaults={
                flag: defaults[flag] for flag in sorted(TRAINING_BUDGET_FLAGS & defaults.keys())
            },
            algorithm_step_defaults={
                flag: defaults[flag] for flag in sorted(ALGORITHM_STEP_FLAGS & defaults.keys())
            },
            data_scale_defaults={
                flag: defaults[flag] for flag in sorted(DATA_SCALE_FLAGS & defaults.keys())
            },
            limit_defaults={
                flag: defaults[flag] for flag in sorted(TRAINING_LIMIT_FLAGS & defaults.keys())
            },
            command_template=_command_template(contract, defaults),
            estimated_train_batches=estimated_batches,
            estimate_basis=estimate_basis,
            resource_band=resource_band,
            required_artifacts=_required_artifacts(contract.entrypoint_kind or "unknown"),
            artifact_candidates=artifacts,
            documented_artifacts=_documented_artifacts(lesson_dir / "README.md"),
            third_party_imports=third_party,
            network_indicators=indicators,
            external_dependency_class=external_class,
            benchmark_classification=benchmark_classification,
            benchmark_mentions=benchmark_mentions,
        )
        specs.append(spec)

    estimated = [spec for spec in specs if spec.estimated_train_batches is not None]
    bands: dict[str, int] = {}
    benchmark_classes: dict[str, int] = {}
    for spec in specs:
        bands[spec.resource_band] = bands.get(spec.resource_band, 0) + 1
        benchmark_classes[spec.benchmark_classification] = (
            benchmark_classes.get(spec.benchmark_classification, 0) + 1
        )
    max_spec = max(estimated, key=lambda spec: spec.estimated_train_batches or 0)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "source_snapshot": source_snapshot,
        "policy": {
            "meaning_of_complete": (
                "default epochs/steps/batch-size/data scale on the offline route; only device, "
                "run-name, and required dataset=fake are injected"
            ),
            "forbidden_overrides": ["--max-*", "epochs", "steps", "batch-size", "data-size"],
            "network_default": (
                "offline environment plus Python AF_INET/AF_INET6 socket blocking; external "
                "subprocess binaries are not sandboxed"
            ),
            "runtime_root": str(default_runtime_root()),
            "execution": "serial; CUDA cache cleared after every lesson",
        },
        "summary": {
            "lessons": len(specs),
            "train_entrypoints": sum(spec.kind == "train" for spec in specs),
            "run_entrypoints": sum(spec.kind == "run" for spec in specs),
            "built_in_offline": sum(spec.offline_mode == "built-in" for spec in specs),
            "explicit_fake": sum(spec.offline_mode == OFFLINE_EXPLICIT_FAKE for spec in specs),
            "external_only": sum(spec.offline_mode == "external-only" for spec in specs),
            "estimated_lessons": len(estimated),
            "unestimated_lessons": len(specs) - len(estimated),
            "estimated_train_batches": sum(spec.estimated_train_batches or 0 for spec in estimated),
            "maximum_estimated_train_batches": max_spec.estimated_train_batches,
            "maximum_estimated_lesson": max_spec.lesson_id,
            "resource_bands": dict(sorted(bands.items())),
            "epoch_default_lessons": sum("--epochs" in spec.cli_defaults for spec in specs),
            "training_limit_flag_lessons": sum(bool(spec.limit_defaults) for spec in specs),
            "non_null_training_limit_defaults": sum(
                value is not None for spec in specs for value in spec.limit_defaults.values()
            ),
            "standard_artifact_train_lessons": sum(
                spec.kind == "train"
                and set(spec.required_artifacts)
                == {
                    "config.json",
                    "metrics.jsonl",
                    "logs/train.log",
                    "checkpoints/checkpoint.pt",
                }
                for spec in specs
            ),
            "benchmark_classifications": dict(sorted(benchmark_classes.items())),
            "benchmark_review_required": sum(
                spec.benchmark_classification == "review-required" for spec in specs
            ),
            "network_indicator_lessons": sum(bool(spec.network_indicators) for spec in specs),
        },
        "lessons": [asdict(spec) | {"lesson_id": spec.lesson_id} for spec in specs],
    }
    stable = json.dumps(
        {"lessons": payload["lessons"], "source_snapshot": source_snapshot},
        sort_keys=True,
        ensure_ascii=False,
    ).encode()
    payload["inventory_sha256"] = hashlib.sha256(stable).hexdigest()
    _assert_source_snapshot(root, source_snapshot)
    return payload


def build_full_command(
    spec: dict[str, Any] | LessonRunSpec,
    *,
    python: str,
    device: str,
    run_name: str,
) -> list[str]:
    template = (
        spec.command_template if isinstance(spec, LessonRunSpec) else spec["command_template"]
    )
    replacements = {"{python}": python, "{device}": device, "{run_name}": run_name}
    command = [replacements.get(token, token) for token in template]
    flags = {token for token in command if token.startswith("--")}
    if any(flag.startswith("--max-") for flag in flags):
        raise ValueError(f"complete command must not contain max-* overrides: {command!r}")
    forbidden = flags - ALLOWED_INJECTED_FLAGS
    if forbidden:
        raise ValueError(f"complete command contains budget overrides: {sorted(forbidden)}")
    return command


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


@contextlib.contextmanager
def _cuda_execution_lock(
    runtime_root: Path,
    *,
    enabled: bool,
    timeout_seconds: float,
    run_id: str,
):
    if not enabled:
        yield None
        return
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("CUDA lock timeout must be positive and finite")
    lock_path = runtime_root / "cuda.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    last_notice = -30.0
    with lock_path.open("a+", encoding="utf-8") as stream:
        while True:
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                elapsed = time.monotonic() - started
                if elapsed >= timeout_seconds:
                    raise TimeoutError(
                        f"timed out after {timeout_seconds:g}s waiting for {lock_path}"
                    ) from None
                if elapsed - last_notice >= 30.0:
                    print(f"waiting for CUDA audit lock {lock_path} ({elapsed:.0f}s)", flush=True)
                    last_notice = elapsed
                time.sleep(min(1.0, timeout_seconds - elapsed))
        stream.seek(0)
        stream.truncate()
        json.dump(
            {"pid": os.getpid(), "run_id": run_id, "acquired_at": _utc_now()},
            stream,
            sort_keys=True,
        )
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
        try:
            yield lock_path
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _resolve_device(requested: str, cuda_index: int) -> tuple[str, dict[str, Any]]:
    if cuda_index < 0:
        raise ValueError("cuda_index must be non-negative")
    if requested == "cpu":
        return "cpu", {"requested": requested, "resolved": "cpu"}
    import torch

    available = torch.cuda.is_available() and cuda_index < torch.cuda.device_count()
    if requested == "cuda" and not available:
        raise ValueError(f"CUDA device index {cuda_index} is unavailable")
    if requested == "auto" and not available:
        return "cpu", {"requested": requested, "resolved": "cpu"}
    properties = torch.cuda.get_device_properties(cuda_index)
    return "cuda", {
        "requested": requested,
        "resolved": "cuda",
        "cuda_index": cuda_index,
        "cuda_name": properties.name,
        "cuda_total_memory_bytes": properties.total_memory,
    }


def _prepare_bootstrap(run_root: Path, *, allow_network: bool) -> Path:
    bootstrap = run_root / "bootstrap"
    bootstrap.mkdir(parents=True, exist_ok=True)
    contents = RUNTIME_SITE_CUSTOMIZE
    if not allow_network:
        contents += OFFLINE_SOCKET_BLOCK
    (bootstrap / "sitecustomize.py").write_text(contents, encoding="utf-8")
    return bootstrap


def _runtime_environment(
    *,
    root: Path,
    run_root: Path,
    case_root: Path,
    tmpdir: Path,
    bootstrap: Path,
    allow_network: bool,
    device: str,
    cuda_index: int,
    cpu_threads: int,
) -> dict[str, str]:
    environment = os.environ.copy()
    python_paths = [str(root)]
    python_paths.insert(0, str(bootstrap))
    if environment.get("PYTHONPATH"):
        python_paths.append(environment["PYTHONPATH"])
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "" if device == "cpu" else str(cuda_index),
            "DLHUB_OUTPUTS_DIR": str(run_root / "outputs"),
            "DLHUB_RUNTIME_AUDIT_GPU_METRICS_DIR": str(case_root / "gpu-metrics"),
            "HF_HOME": str(case_root / "cache" / "hf"),
            "MKL_NUM_THREADS": str(cpu_threads),
            "MPLBACKEND": "Agg",
            "MPLCONFIGDIR": str(case_root / "cache" / "mpl"),
            "NUMEXPR_NUM_THREADS": str(cpu_threads),
            "OMP_NUM_THREADS": str(cpu_threads),
            "OPENBLAS_NUM_THREADS": str(cpu_threads),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "PYTHONPATH": os.pathsep.join(python_paths),
            "TMPDIR": str(tmpdir),
            "TOKENIZERS_PARALLELISM": "false",
            "TORCH_HOME": str(case_root / "cache" / "torch"),
            "XDG_CACHE_HOME": str(case_root / "cache" / "xdg"),
        }
    )
    if not allow_network:
        environment.update(
            {
                "DIFFUSERS_OFFLINE": "1",
                "HF_DATASETS_OFFLINE": "1",
                "HF_HUB_OFFLINE": "1",
                "PIP_NO_INDEX": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "WANDB_MODE": "offline",
            }
        )
    return environment


@contextlib.contextmanager
def _short_tmpdir(case_root: Path):
    """Expose the case-local temp tree through a short AF_UNIX-safe path."""

    target = (case_root / "tmp").resolve()
    target.mkdir(parents=True, exist_ok=True)
    link: Path | None = None
    for _ in range(100):
        candidate = Path("/tmp") / f"dlh-{os.getpid()}-{secrets.token_hex(4)}"
        try:
            candidate.symlink_to(target, target_is_directory=True)
        except FileExistsError:
            continue
        link = candidate
        break
    if link is None:
        raise RuntimeError("could not allocate a short runtime-audit TMPDIR symlink")

    try:
        yield link
    finally:
        if link.is_symlink():
            link.unlink()


def _tail_file(path: Path, limit: int = 4_000) -> str:
    if not path.is_file():
        return ""
    with path.open("rb") as stream:
        stream.seek(0, os.SEEK_END)
        size = stream.tell()
        stream.seek(max(0, size - limit))
        return stream.read().decode("utf-8", errors="replace")


def _collect_gpu_metrics(case_root: Path, *, device: str) -> dict[str, Any]:
    metrics_dir = case_root / "gpu-metrics"
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    if metrics_dir.is_dir():
        for path in sorted(metrics_dir.glob("*.json")):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                errors.append(f"{path.name}: {exc}")
                continue
            if isinstance(record, dict):
                records.append(record)
    allocated = [
        value
        for record in records
        for value in record.get("devices", {}).values()
        if not isinstance(value, bool) and isinstance(value, int) and value >= 0
    ]
    return {
        "metric": "torch.cuda.max_memory_allocated",
        "scope": "peak live tensor bytes tracked by the PyTorch allocator in child processes",
        "peak_bytes": max(allocated, default=None),
        "process_records": records,
        "errors": errors,
        "available": bool(records),
        "unavailable_reason": (
            "CPU execution"
            if device == "cpu"
            else None
            if records
            else "child exited without emitting a PyTorch CUDA allocator record"
        ),
    }


def _gpu_measurement_error(
    spec: dict[str, Any],
    *,
    device: str,
    returncode: int | None,
    gpu_memory: dict[str, Any],
    artifact_validation: Mapping[str, Any] | None = None,
) -> str | None:
    if returncode != 0 or device != "cuda" or spec["kind"] != "train":
        return None
    if (
        artifact_validation is not None
        and artifact_validation.get("ok") is True
        and artifact_validation.get("model_free") is True
    ):
        return None
    peak_bytes = gpu_memory["peak_bytes"]
    if (
        gpu_memory["available"]
        and not isinstance(peak_bytes, bool)
        and isinstance(peak_bytes, int)
        and peak_bytes > 0
    ):
        return None
    return "missing positive integer child PyTorch CUDA peak-memory measurement"


def _terminate_process_group(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=5)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def _iter_numbers(value: Any):
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


def _validate_artifacts(spec: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    if spec["kind"] == "run":
        return {
            "ok": True,
            "required": [],
            "missing": [],
            "files": [],
            "model_free": False,
        }
    files = (
        sorted(
            path.relative_to(run_dir).as_posix() for path in run_dir.rglob("*") if path.is_file()
        )
        if run_dir.is_dir()
        else []
    )
    required = list(spec["required_artifacts"])
    missing = [relative for relative in required if not (run_dir / relative).is_file()]
    errors: list[str] = []
    if not run_dir.is_dir():
        errors.append(f"missing run directory: {run_dir}")
    for relative in required:
        path = run_dir / relative
        if path.is_file() and path.stat().st_size == 0:
            errors.append(f"empty artifact: {relative}")
    config_path = run_dir / "config.json"
    if config_path.is_file():
        try:
            json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"invalid config.json: {exc}")
    metrics_path = run_dir / "metrics.jsonl"
    metric_records = 0
    if metrics_path.is_file():
        for line_number, line in enumerate(
            metrics_path.read_text(encoding="utf-8").splitlines(), 1
        ):
            if not line.strip():
                continue
            metric_records += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"metrics.jsonl:{line_number}: {exc}")
                continue
            values = list(_iter_numbers(record))
            if any(not math.isfinite(value) for value in values):
                errors.append(f"metrics.jsonl:{line_number}: non-finite numeric value")
        if metric_records == 0:
            errors.append("metrics.jsonl has no records")
    checkpoint_path = run_dir / "checkpoints" / "checkpoint.pt"
    checkpoint_keys: list[str] = []
    model_free = False
    if checkpoint_path.is_file():
        try:
            import torch

            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            if not isinstance(checkpoint, Mapping):
                errors.append("checkpoint.pt is not a mapping")
            else:
                checkpoint_keys = sorted(str(key) for key in checkpoint)
                model_state = checkpoint.get("model_state")
                extra = checkpoint.get("extra")
                if (
                    isinstance(model_state, Mapping)
                    and not model_state
                    and isinstance(extra, Mapping)
                    and extra.get("model_free") is True
                ):
                    model_free = True
                elif not isinstance(model_state, Mapping) or not model_state:
                    errors.append("checkpoint.pt has no non-empty model_state")
            del checkpoint
        except Exception as exc:
            errors.append(f"checkpoint.pt cannot be safely loaded: {type(exc).__name__}: {exc}")
    return {
        "ok": not missing and not errors,
        "required": required,
        "missing": missing,
        "errors": errors,
        "files": files,
        "metric_records": metric_records,
        "checkpoint_keys": checkpoint_keys,
        "model_free": model_free,
    }


def _clear_cuda_cache(device: str) -> None:
    if device != "cuda":
        return
    try:
        import torch

        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    except Exception:
        pass


def _run_one(
    spec: dict[str, Any],
    *,
    root: Path,
    run_root: Path,
    run_name: str,
    attempt: int,
    device: str,
    cuda_index: int,
    cpu_threads: int,
    timeout_seconds: float,
    bootstrap: Path,
    allow_network: bool,
) -> dict[str, Any]:
    lesson_id = spec["lesson_id"]
    case_root = run_root / "lessons" / lesson_id.replace("/", "__") / f"attempt-{attempt:03d}"
    workdir = case_root / "cwd"
    for directory in (
        workdir,
        case_root / "cache",
        case_root / "gpu-metrics",
        case_root / "tmp",
    ):
        directory.mkdir(parents=True, exist_ok=True)
    stdout_path = case_root / "stdout.log"
    stderr_path = case_root / "stderr.log"
    command = build_full_command(
        spec,
        python=sys.executable,
        device=device,
        run_name=run_name,
    )
    started_at = _utc_now()
    started = time.monotonic()
    state = "failed_exit"
    error: str | None = None
    returncode: int | None = None
    interrupted = False
    with _short_tmpdir(case_root) as tmpdir:
        environment = _runtime_environment(
            root=root,
            run_root=run_root,
            case_root=case_root,
            tmpdir=tmpdir,
            bootstrap=bootstrap,
            allow_network=allow_network,
            device=device,
            cuda_index=cuda_index,
            cpu_threads=cpu_threads,
        )
        with (
            stdout_path.open("w", encoding="utf-8") as stdout,
            stderr_path.open("w", encoding="utf-8") as stderr,
        ):
            try:
                process = subprocess.Popen(
                    command,
                    cwd=workdir,
                    env=environment,
                    stdout=stdout,
                    stderr=stderr,
                    text=True,
                    start_new_session=True,
                )
            except OSError as exc:
                state = "failed_start"
                error = f"could not start entrypoint: {type(exc).__name__}: {exc}"
            else:
                try:
                    returncode = process.wait(timeout=timeout_seconds)
                except subprocess.TimeoutExpired:
                    _terminate_process_group(process)
                    returncode = process.returncode
                    state = "timed_out"
                    error = f"exceeded {timeout_seconds:g}s timeout"
                except KeyboardInterrupt:
                    _terminate_process_group(process)
                    returncode = process.returncode
                    state = "interrupted"
                    error = "runner interrupted"
                    interrupted = True
                else:
                    state = "passed" if returncode == 0 else "failed_exit"
                    if returncode != 0:
                        error = f"entrypoint exited with {returncode}"

    run_dir = run_root / "outputs" / spec["track"] / spec["lesson"] / run_name
    artifact_validation: dict[str, Any] | None = None
    if state == "passed":
        artifact_validation = _validate_artifacts(spec, run_dir)
        if not artifact_validation["ok"]:
            state = "failed_validation"
            error = "entrypoint exited 0 but artifact validation failed"
    gpu_memory = _collect_gpu_metrics(case_root, device=device)
    runtime_validation_errors: list[str] = []
    gpu_error = _gpu_measurement_error(
        spec,
        device=device,
        returncode=returncode,
        gpu_memory=gpu_memory,
        artifact_validation=artifact_validation,
    )
    if gpu_error:
        runtime_validation_errors.append(gpu_error)
        state = "failed_validation"
        error = "entrypoint exited 0 but runtime validation failed"
    _clear_cuda_cache(device)
    result = {
        "state": state,
        "attempt": attempt,
        "run_name": run_name,
        "command": command,
        "started_at": started_at,
        "finished_at": _utc_now(),
        "elapsed_seconds": time.monotonic() - started,
        "returncode": returncode,
        "error": error,
        "case_root": str(case_root),
        "run_dir": str(run_dir),
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
        "stdout_tail": _tail_file(stdout_path),
        "stderr_tail": _tail_file(stderr_path),
        "artifact_validation": artifact_validation,
        "gpu_memory": gpu_memory,
        "runtime_validation_errors": runtime_validation_errors,
        "interrupted": interrupted,
    }
    return result


def _select_lessons(
    manifest: dict[str, Any],
    *,
    run_all: bool,
    lessons: list[str],
    tracks: list[str],
    limit: int | None,
) -> list[dict[str, Any]]:
    if not run_all and not lessons and not tracks:
        raise ValueError("choose --all, --lesson, or --track; full execution is never implicit")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    wanted_lessons = set(lessons)
    wanted_tracks = set(tracks)
    selected = [
        spec
        for spec in manifest["lessons"]
        if run_all
        or spec["lesson_id"] in wanted_lessons
        or spec["module"] in wanted_lessons
        or spec["track"] in wanted_tracks
    ]
    unknown = wanted_lessons - {
        value for spec in manifest["lessons"] for value in (spec["lesson_id"], spec["module"])
    }
    if unknown:
        raise ValueError(f"unknown lessons: {sorted(unknown)}")
    if not selected:
        raise ValueError("selection matched no lessons")
    if limit is not None:
        selected = selected[:limit]
    return selected


def _report_summary(report: dict[str, Any]) -> dict[str, int]:
    states: dict[str, int] = {}
    for result in report["lessons"].values():
        state = result["state"]
        states[state] = states.get(state, 0) + 1
    states["total"] = len(report["lessons"])
    return dict(sorted(states.items()))


def _attempt_run_name(base_run_name: str, lesson_id: str, attempt: int) -> str:
    lesson_hash = hashlib.sha256(lesson_id.encode()).hexdigest()[:12]
    return f"{base_run_name}-{lesson_hash}-a{attempt:03d}"


def run_manifest_selection(
    manifest: dict[str, Any],
    selected: list[dict[str, Any]],
    *,
    root: Path,
    runtime_root: Path,
    report_path: Path,
    run_id: str,
    device: str,
    device_details: dict[str, Any],
    cuda_index: int,
    cpu_threads: int,
    timeout_seconds: float,
    cuda_lock_timeout_seconds: float,
    allow_network: bool,
    resume: bool,
    retry_failed: bool,
) -> dict[str, Any]:
    if isinstance(cpu_threads, bool) or not isinstance(cpu_threads, int) or cpu_threads < 1:
        raise ValueError("cpu_threads must be positive")
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, int | float)
        or not math.isfinite(timeout_seconds)
        or timeout_seconds <= 0
    ):
        raise ValueError("timeout must be positive and finite")
    if (
        isinstance(cuda_lock_timeout_seconds, bool)
        or not isinstance(cuda_lock_timeout_seconds, int | float)
        or not math.isfinite(cuda_lock_timeout_seconds)
        or cuda_lock_timeout_seconds <= 0
    ):
        raise ValueError("CUDA lock timeout must be positive and finite")
    runtime_root = runtime_root.resolve()
    report_path = report_path.resolve()
    if runtime_root != report_path and runtime_root not in report_path.parents:
        raise ValueError(f"report must stay under {runtime_root}")
    run_root = report_path.parent
    run_root.mkdir(parents=True, exist_ok=True)
    run_name = f"full-{run_id}"

    if resume:
        if not report_path.is_file():
            raise ValueError(f"resume report does not exist: {report_path}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("inventory_sha256") != manifest["inventory_sha256"]:
            raise ValueError("lesson inventory changed since the report was created")
        selected_ids = set(report["lessons"])
        selected = [spec for spec in manifest["lessons"] if spec["lesson_id"] in selected_ids]
        run_name = report["config"]["run_name"]
        device = report["config"]["device"]
        cuda_index = report["config"]["cuda_index"]
        cpu_threads = report["config"]["cpu_threads"]
        timeout_seconds = report["config"]["timeout_seconds"]
        cuda_lock_timeout_seconds = report["config"].get(
            "cuda_lock_timeout_seconds", cuda_lock_timeout_seconds
        )
        allow_network = report["config"]["allow_network"]
        for entry in report["lessons"].values():
            if entry["state"] == "running":
                entry["state"] = "interrupted"
    else:
        if report_path.exists():
            raise ValueError(f"report already exists; pass --resume: {report_path}")
        report = {
            "schema_version": SCHEMA_VERSION,
            "inventory_sha256": manifest["inventory_sha256"],
            "source_snapshot": manifest["source_snapshot"],
            "created_at": _utc_now(),
            "updated_at": _utc_now(),
            "status": "running",
            "runtime_root": str(runtime_root),
            "run_root": str(run_root),
            "config": {
                "run_id": run_id,
                "run_name": run_name,
                "device": device,
                "device_details": device_details,
                "cuda_index": cuda_index,
                "cpu_threads": cpu_threads,
                "timeout_seconds": timeout_seconds,
                "cuda_lock_timeout_seconds": cuda_lock_timeout_seconds,
                "allow_network": allow_network,
                "network_policy": (
                    "allowed" if allow_network else "offline env + Python AF_INET/AF_INET6 blocked"
                ),
                "serial": True,
            },
            "lessons": {
                spec["lesson_id"]: {
                    "state": "pending",
                    "attempts": 0,
                    "attempt_history": [],
                    "estimated_train_batches": spec["estimated_train_batches"],
                }
                for spec in selected
            },
            "summary": {},
        }
    _assert_source_snapshot(root, manifest["source_snapshot"])
    _atomic_write_json(run_root / "manifest.json", manifest)
    bootstrap = _prepare_bootstrap(run_root, allow_network=allow_network)
    _atomic_write_json(report_path, report)

    interrupted = False
    source_changed = False
    try:
        with _cuda_execution_lock(
            runtime_root,
            enabled=device == "cuda",
            timeout_seconds=cuda_lock_timeout_seconds,
            run_id=run_id,
        ):
            for position, spec in enumerate(selected, 1):
                lesson_id = spec["lesson_id"]
                entry = report["lessons"][lesson_id]
                if entry["state"] == "passed":
                    continue
                if entry["state"] in TERMINAL_FAILURE_STATES and not retry_failed:
                    continue
                try:
                    _assert_source_snapshot(root, manifest["source_snapshot"])
                except RuntimeError as exc:
                    source_changed = True
                    report["source_change_error"] = str(exc)
                    break
                attempt = int(entry.get("attempts", 0)) + 1
                attempt_run_name = _attempt_run_name(run_name, lesson_id, attempt)
                entry.update(
                    {
                        "state": "running",
                        "attempts": attempt,
                        "position": position,
                    }
                )
                report["updated_at"] = _utc_now()
                report["summary"] = _report_summary(report)
                _atomic_write_json(report_path, report)
                print(f"[{position}/{len(selected)}] {lesson_id}: starting", flush=True)
                result = _run_one(
                    spec,
                    root=root,
                    run_root=run_root,
                    run_name=attempt_run_name,
                    attempt=attempt,
                    device=device,
                    cuda_index=cuda_index,
                    cpu_threads=cpu_threads,
                    timeout_seconds=timeout_seconds,
                    bootstrap=bootstrap,
                    allow_network=allow_network,
                )
                try:
                    _assert_source_snapshot(root, manifest["source_snapshot"])
                except RuntimeError as exc:
                    result["source_snapshot_stable"] = False
                    result["state"] = "invalidated_source_change"
                    result["error"] = str(exc)
                    source_changed = True
                    report["source_change_error"] = str(exc)
                else:
                    result["source_snapshot_stable"] = True
                entry.setdefault("attempt_history", []).append(result)
                entry.update(result)
                report["updated_at"] = _utc_now()
                report["summary"] = _report_summary(report)
                _atomic_write_json(report_path, report)
                print(
                    f"[{position}/{len(selected)}] {lesson_id}: {entry['state']} "
                    f"({entry['elapsed_seconds']:.2f}s)",
                    flush=True,
                )
                if entry.get("interrupted"):
                    interrupted = True
                    break
                if source_changed:
                    break
    except TimeoutError as exc:
        report["cuda_lock_error"] = str(exc)
        report["status"] = "cuda_lock_timeout"
        report["updated_at"] = _utc_now()
        report["summary"] = _report_summary(report)
        _atomic_write_json(report_path, report)
        return report

    summary = _report_summary(report)
    report["summary"] = summary
    if source_changed:
        report["status"] = "source_changed"
    elif interrupted:
        report["status"] = "interrupted"
    elif summary.get("pending", 0) or summary.get("running", 0):
        report["status"] = "incomplete"
    elif any(summary.get(state, 0) for state in TERMINAL_FAILURE_STATES):
        report["status"] = "finished_with_failures"
    else:
        report["status"] = "complete"
    report["updated_at"] = _utc_now()
    _atomic_write_json(report_path, report)
    return report


def _new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _safe_run_id(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", value):
        raise ValueError("run-id must contain 1-64 letters, digits, dots, underscores, or hyphens")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    inventory_parser = subparsers.add_parser("inventory", help="Write the 339-lesson manifest.")
    inventory_parser.add_argument(
        "--output",
        type=Path,
        default=default_runtime_root() / "lesson-manifest.json",
    )
    inventory_parser.add_argument("--json", action="store_true", help="Also print JSON to stdout.")

    run_parser = subparsers.add_parser("run", help="Run selected complete default-budget lessons.")
    run_parser.add_argument("--all", action="store_true", help="Select all 339 lessons explicitly.")
    run_parser.add_argument("--lesson", action="append", default=[], help="track/lesson or module.")
    run_parser.add_argument("--track", action="append", default=[], help="Select one track.")
    run_parser.add_argument("--limit", type=int, help="Limit a deterministic selection.")
    run_parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    run_parser.add_argument("--cuda-index", type=int, default=0)
    run_parser.add_argument("--cpu-threads", type=int, default=4)
    run_parser.add_argument("--timeout", type=float, default=3_600.0)
    run_parser.add_argument("--cuda-lock-timeout", type=float, default=600.0)
    run_parser.add_argument("--run-id", default=None)
    run_parser.add_argument("--resume", type=Path, help="Resume an existing report.json.")
    run_parser.add_argument("--retry-failed", action="store_true")
    run_parser.add_argument(
        "--allow-network",
        action="store_true",
        help="Disable the default offline flags and Python socket block.",
    )
    args = parser.parse_args(argv)

    root = repo_root()
    runtime_root = default_runtime_root()
    try:
        manifest = build_lesson_manifest(root)
    except (OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    if args.command == "inventory":
        output = args.output.resolve()
        runtime_root_resolved = runtime_root.resolve()
        if output != runtime_root_resolved and runtime_root_resolved not in output.parents:
            parser.error(f"inventory output must stay under {runtime_root_resolved}")
        _atomic_write_json(output, manifest)
        if args.json:
            print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
        else:
            summary = manifest["summary"]
            print(f"lesson full-run manifest: {summary['lessons']} lessons -> {output}")
            print(
                f"- estimated: {summary['estimated_lessons']} lessons / "
                f"{summary['estimated_train_batches']} default train batches"
            )
            print(
                f"- maximum: {summary['maximum_estimated_lesson']} / "
                f"{summary['maximum_estimated_train_batches']} batches"
            )
        return 0

    try:
        if args.resume:
            report_path = args.resume.resolve()
            runtime_resolved = runtime_root.resolve()
            if runtime_resolved not in report_path.parents:
                raise ValueError(f"resume report must stay under {runtime_resolved}")
            existing = json.loads(report_path.read_text(encoding="utf-8"))
            run_id = existing["config"]["run_id"]
            selected_ids = list(existing["lessons"])
            selected = _select_lessons(
                manifest,
                run_all=False,
                lessons=selected_ids,
                tracks=[],
                limit=None,
            )
        else:
            run_id = _safe_run_id(args.run_id or _new_run_id())
            selected = _select_lessons(
                manifest,
                run_all=args.all,
                lessons=args.lesson,
                tracks=args.track,
                limit=args.limit,
            )
            report_path = runtime_root / "runs" / run_id / "report.json"
        requested_device = args.device
        requested_cuda_index = args.cuda_index
        if args.resume:
            requested_device = existing["config"]["device"]
            requested_cuda_index = existing["config"]["cuda_index"]
        device, device_details = _resolve_device(requested_device, requested_cuda_index)
        report = run_manifest_selection(
            manifest,
            selected,
            root=root,
            runtime_root=runtime_root,
            report_path=report_path,
            run_id=run_id,
            device=device,
            device_details=device_details,
            cuda_index=requested_cuda_index,
            cpu_threads=args.cpu_threads,
            timeout_seconds=args.timeout,
            cuda_lock_timeout_seconds=args.cuda_lock_timeout,
            allow_network=args.allow_network,
            resume=bool(args.resume),
            retry_failed=args.retry_failed,
        )
    except (KeyError, OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(json.dumps({"report": str(report_path), "status": report["status"], **report["summary"]}))
    if report["status"] == "interrupted":
        return 130
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
