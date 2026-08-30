"""Exercise DL-Hub's real device and artifact-storage runtime boundaries.

The check intentionally performs work instead of inferring support from names:
tensor math, a tiny optimizer step, safe checkpoint reload, concurrent atomic
replacement, and concurrent JSONL appends all have to complete successfully.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import tempfile
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


_EVIDENCE_FILES = (
    "scripts/platform_runtime_check.py",
    "dlhub/_atomic.py",
    "dlhub/config.py",
    "dlhub/checkpoint.py",
    "dlhub/device.py",
)


def _git_head_without_cli() -> str:
    dot_git = REPO_ROOT / ".git"
    if dot_git.is_file():
        pointer = dot_git.read_text(encoding="utf-8").strip()
        if not pointer.startswith("gitdir: "):
            return "unavailable"
        git_directory = (REPO_ROOT / pointer.removeprefix("gitdir: ")).resolve()
    else:
        git_directory = dot_git

    head_path = git_directory / "HEAD"
    try:
        head = head_path.read_text(encoding="utf-8").strip()
    except OSError:
        return "unavailable"
    if not head.startswith("ref: "):
        return head

    reference = head.removeprefix("ref: ")
    try:
        return (git_directory / reference).read_text(encoding="utf-8").strip()
    except OSError:
        pass

    try:
        packed_refs = (git_directory / "packed-refs").read_text(encoding="utf-8")
    except OSError:
        return "unavailable"
    for line in packed_refs.splitlines():
        if line.startswith(("#", "^")):
            continue
        fields = line.split(" ", 1)
        if len(fields) == 2 and fields[1] == reference:
            return fields[0]
    return "unavailable"


def _source_evidence() -> dict[str, Any]:
    hashes = {
        relative_path: hashlib.sha256((REPO_ROOT / relative_path).read_bytes()).hexdigest()
        for relative_path in _EVIDENCE_FILES
    }
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        dirty: bool | None = bool(status.strip())
    except (OSError, subprocess.CalledProcessError):
        head = os.environ.get("GITHUB_SHA") or _git_head_without_cli()
        dirty = None

    return {
        "git_head": head,
        "git_worktree_dirty": dirty,
        "sha256": hashes,
    }


def _ci_evidence() -> dict[str, str | None]:
    return {
        "github_sha": os.environ.get("GITHUB_SHA"),
        "github_run_id": os.environ.get("GITHUB_RUN_ID"),
        "runner_os": os.environ.get("RUNNER_OS"),
        "runner_arch": os.environ.get("RUNNER_ARCH"),
        "image_os": os.environ.get("ImageOS"),
        "image_version": os.environ.get("ImageVersion"),
    }


def _decode_mount_field(value: str) -> str:
    """Decode the octal escapes used by Linux ``/proc/*/mountinfo``."""

    return re.sub(r"\\([0-7]{3})", lambda match: chr(int(match.group(1), 8)), value)


def _linux_device_id(major: int, minor: int) -> int:
    """Encode a Linux major/minor pair, including in parser tests on Windows."""

    make_device = getattr(os, "makedev", None)
    if make_device is not None:
        return int(make_device(major, minor))
    return (major << 32) | minor


def _linux_mount_for_path(
    mountinfo: str,
    path: str | Path,
    *,
    device_id: int | None = None,
) -> dict[str, str] | None:
    """Return the longest Linux mountinfo entry containing ``path``."""

    # ``mountinfo`` always uses POSIX paths.  Keep those semantics when the
    # pure parser is contract-tested on Windows instead of letting
    # ``WindowsPath.resolve()`` turn ``/mnt/...`` into a drive-relative path.
    if os.name == "nt":
        target = os.fspath(path).replace("\\", "/")
    else:
        resolved_path = Path(path).resolve()
        target = str(resolved_path)
        if device_id is None:
            try:
                device_id = resolved_path.stat().st_dev
            except OSError:
                device_id = None
    best: dict[str, str] | None = None
    best_score = (-1, -1, -1)

    for line_number, raw_line in enumerate(mountinfo.splitlines()):
        try:
            before_separator, after_separator = raw_line.split(" - ", 1)
        except ValueError:
            continue
        mount_fields = before_separator.split()
        filesystem_fields = after_separator.split()
        if len(mount_fields) < 6 or len(filesystem_fields) < 2:
            continue

        mount_point = _decode_mount_field(mount_fields[4])
        prefix = mount_point.rstrip("/")
        contains_target = mount_point == "/" or target == mount_point
        if prefix:
            contains_target = contains_target or target.startswith(f"{prefix}/")
        if not contains_target:
            continue

        candidate = {
            "type": filesystem_fields[0].lower(),
            "mount_point": mount_point,
            "source": _decode_mount_field(filesystem_fields[1]),
            "device": mount_fields[2],
        }
        candidate_device: int | None = None
        if device_id is not None:
            try:
                major_text, minor_text = mount_fields[2].split(":", 1)
                candidate_device = _linux_device_id(int(major_text), int(minor_text))
            except (ValueError, OSError):
                candidate_device = None
        device_matches = int(device_id is not None and candidate_device == device_id)
        # Device identity is decisive for stacked mounts: the visible path's
        # st_dev matches the top mount, while an obscured bind mount does not.
        # Mount IDs are intentionally not used because Linux recycles them.
        score = (device_matches, len(mount_point), line_number)
        if score > best_score:
            best = candidate
            best_score = score

    return best


def _windows_volume_for_path(path: Path) -> dict[str, str]:
    """Resolve a Windows volume and filesystem with the native Win32 API."""

    import ctypes
    from ctypes import wintypes

    resolved = str(path.resolve())
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    volume_path = ctypes.create_unicode_buffer(261)
    filesystem_name = ctypes.create_unicode_buffer(261)
    serial_number = wintypes.DWORD()
    maximum_component_length = wintypes.DWORD()
    filesystem_flags = wintypes.DWORD()

    kernel32.GetVolumePathNameW.argtypes = (
        wintypes.LPCWSTR,
        wintypes.LPWSTR,
        wintypes.DWORD,
    )
    kernel32.GetVolumePathNameW.restype = wintypes.BOOL
    if not kernel32.GetVolumePathNameW(resolved, volume_path, len(volume_path)):
        error = ctypes.get_last_error()
        raise OSError(error, ctypes.FormatError(error), resolved)

    kernel32.GetVolumeInformationW.argtypes = (
        wintypes.LPCWSTR,
        wintypes.LPWSTR,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
        ctypes.POINTER(wintypes.DWORD),
        ctypes.POINTER(wintypes.DWORD),
        wintypes.LPWSTR,
        wintypes.DWORD,
    )
    kernel32.GetVolumeInformationW.restype = wintypes.BOOL
    if not kernel32.GetVolumeInformationW(
        volume_path.value,
        None,
        0,
        ctypes.byref(serial_number),
        ctypes.byref(maximum_component_length),
        ctypes.byref(filesystem_flags),
        filesystem_name,
        len(filesystem_name),
    ):
        error = ctypes.get_last_error()
        raise OSError(error, ctypes.FormatError(error), volume_path.value)

    if not filesystem_name.value:
        raise RuntimeError(f"Windows returned no filesystem name for {volume_path.value}")
    return {
        "type": filesystem_name.value.lower(),
        "mount_point": volume_path.value,
        "source": volume_path.value,
        "evidence": "Win32 GetVolumePathNameW + GetVolumeInformationW",
    }


def _filesystem_evidence(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    if sys.platform.startswith("linux"):
        try:
            mountinfo = Path("/proc/self/mountinfo").read_text(encoding="utf-8")
        except OSError as exc:
            return {
                "type": "unverified",
                "path": str(resolved),
                "evidence": f"could not read /proc/self/mountinfo: {exc}",
            }

        entry = _linux_mount_for_path(mountinfo, resolved)
        if entry is None:
            return {
                "type": "unverified",
                "path": str(resolved),
                "evidence": "no containing entry in /proc/self/mountinfo",
            }
        return {
            **entry,
            "path": str(resolved),
            "evidence": "/proc/self/mountinfo",
        }

    if os.name == "nt":
        return {**_windows_volume_for_path(resolved), "path": str(resolved)}

    return {
        "type": "unverified",
        "path": str(resolved),
        "evidence": f"filesystem type detection is not implemented for {sys.platform}",
    }


def _require_filesystem(filesystem: dict[str, str], required: str | None) -> None:
    if required is None:
        return

    expected = required.strip().lower()
    actual = filesystem["type"].lower()
    matches = actual == expected or (expected == "nfs" and actual in {"nfs", "nfs4"})
    if not matches:
        raise RuntimeError(
            f"required filesystem {expected!r}, but the work directory is on {actual!r}; "
            f"evidence: {filesystem['evidence']}"
        )


def _resolve_requirements(
    requirements: tuple[str, ...],
    *,
    legacy_device: str | None,
    legacy_filesystem: str | None,
    filesystem: dict[str, str],
) -> str | None:
    requested = set(requirements)
    if "windows" in requested:
        if platform.system() != "Windows":
            raise RuntimeError(
                f"required platform 'windows', but platform.system() is {platform.system()!r}"
            )
        if filesystem["type"] == "unverified":
            raise RuntimeError("Windows filesystem type could not be verified with the Win32 API")

    filesystem_requirement = "nfs" if "nfs" in requested else legacy_filesystem
    _require_filesystem(filesystem, filesystem_requirement)

    accelerated = requested.intersection({"cuda", "mps"})
    if len(accelerated) > 1:
        raise ValueError("--require cuda and --require mps are mutually exclusive")
    unified_device = next(iter(accelerated), None)
    if legacy_device is not None and unified_device is not None and legacy_device != unified_device:
        raise ValueError(
            f"conflicting device requirements: {legacy_device!r} and {unified_device!r}"
        )
    return unified_device or legacy_device


def _remove_runtime_directory(directory: Path, timeout_seconds: float) -> dict[str, Any]:
    import shutil

    timeout_enforced = False
    previous_handler: Any = None
    previous_timer: tuple[float, float] | None = None
    if os.name != "nt":
        import signal

        def handle_timeout(_signum: int, _frame: Any) -> None:
            raise TimeoutError

        try:
            previous_handler = signal.signal(signal.SIGALRM, handle_timeout)
            previous_timer = signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
            timeout_enforced = True
        except (AttributeError, ValueError):
            timeout_enforced = False

    try:
        shutil.rmtree(directory)
    except TimeoutError as exc:
        raise RuntimeError(
            f"workspace cleanup exceeded {timeout_seconds:g}s; retained partial workspace at "
            f"{directory}"
        ) from exc
    finally:
        if timeout_enforced:
            import signal

            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous_handler)
            if previous_timer is not None and previous_timer[0] > 0:
                signal.setitimer(signal.ITIMER_REAL, *previous_timer)

    return {
        "status": "removed",
        "timeout_seconds": timeout_seconds,
        "timeout_enforced": timeout_enforced,
    }


def _worker_environment() -> dict[str, str]:
    environment = os.environ.copy()
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        str(REPO_ROOT) if not existing else f"{REPO_ROOT}{os.pathsep}{existing}"
    )
    return environment


def _launch_worker(
    worker_type: str,
    path: Path,
    worker_id: int,
    iterations: int,
) -> subprocess.Popen[str]:
    return subprocess.Popen(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--_worker",
            worker_type,
            "--_path",
            str(path),
            "--_worker-id",
            str(worker_id),
            "--_iterations",
            str(iterations),
        ],
        cwd=REPO_ROOT,
        env=_worker_environment(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _collect_workers(processes: list[subprocess.Popen[str]], label: str) -> None:
    failures: list[str] = []
    for process in processes:
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            failures.append(
                f"pid={process.pid} exit={process.returncode} "
                f"stdout={stdout[-1000:]!r} stderr={stderr[-2000:]!r}"
            )
    if failures:
        raise RuntimeError(f"{label} worker failure: {'; '.join(failures)}")


def _run_atomic_worker(path: Path, worker_id: int, iterations: int) -> None:
    from dlhub.config import write_json

    for sequence in range(iterations):
        write_json(
            path,
            {
                "payload": "atomic-replacement-proof-" * 256,
                "sequence": sequence,
                "worker": worker_id,
            },
        )
        time.sleep(0.001 * ((worker_id + sequence) % 3))


def _run_failing_writer(path: Path) -> None:
    from dlhub._atomic import atomic_write

    def fail_after_partial_write(handle: Any) -> None:
        handle.write(b"intentionally incomplete replacement")
        raise RuntimeError("intentional writer failure after partial write")

    atomic_write(path, fail_after_partial_write)


def _run_jsonl_worker(path: Path, worker_id: int, iterations: int) -> None:
    from dlhub.config import append_jsonl

    for sequence in range(iterations):
        append_jsonl(
            path,
            {
                "payload": "concurrent-append-proof-" * 16,
                "sequence": sequence,
                "worker": worker_id,
            },
        )
        time.sleep(0.001 * ((worker_id + sequence) % 3))


def _run_interrupted_write_check(directory: Path) -> dict[str, Any]:
    from dlhub.config import write_json

    target = directory / "last-good.json"
    write_json(target, {"generation": "last-good"})
    last_good = target.read_bytes()

    process = _launch_worker("failing", target, 0, 1)
    stdout, stderr = process.communicate()
    if process.returncode == 0:
        raise RuntimeError("intentional failing writer unexpectedly succeeded")
    if target.read_bytes() != last_good:
        raise RuntimeError("failing writer damaged the last known-good file")

    temporary_files = sorted(path.name for path in directory.glob(".*.tmp"))
    if temporary_files:
        raise RuntimeError(f"failing writer leaked temporary files: {temporary_files}")

    return {
        "status": "passed",
        "worker_exit_code": process.returncode,
        "old_file": "preserved",
        "temporary_files": "cleaned",
        "failure_observed": "intentional writer failure" in stderr,
        "worker_stdout_bytes": len(stdout.encode()),
    }


def _read_atomic_json(
    target: Path,
    *,
    retry_permission_denied: bool | None = None,
    timeout_seconds: float = 2.0,
) -> tuple[Any, int]:
    """Read JSON, retrying only Windows' transient replace sharing window."""

    retry_permission_denied = (
        os.name == "nt" if retry_permission_denied is None else retry_permission_denied
    )
    deadline = time.monotonic() + timeout_seconds
    permission_retries = 0
    while True:
        try:
            return json.loads(target.read_text(encoding="utf-8")), permission_retries
        except PermissionError:
            if not retry_permission_denied or time.monotonic() >= deadline:
                raise
            permission_retries += 1
            time.sleep(0.001)


def _run_atomic_replace_check(
    directory: Path,
    *,
    writers: int,
    writes_per_worker: int,
) -> dict[str, Any]:
    from dlhub.config import write_json

    target = directory / "runtime-state.json"
    write_json(target, {"payload": "initial", "sequence": -1, "worker": -1})
    processes = [
        _launch_worker("atomic", target, worker_id, writes_per_worker)
        for worker_id in range(writers)
    ]

    observations = 0
    reader_access_retries = 0
    invalid_observations: list[str] = []
    while any(process.poll() is None for process in processes):
        try:
            payload, access_retries = _read_atomic_json(target)
            reader_access_retries += access_retries
            if not isinstance(payload, dict) or not {
                "payload",
                "sequence",
                "worker",
            }.issubset(payload):
                invalid_observations.append(repr(payload))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            invalid_observations.append(f"{type(exc).__name__}: {exc}")
        observations += 1
        time.sleep(0.001)

    _collect_workers(processes, "atomic replacement")
    if invalid_observations:
        raise RuntimeError(
            "concurrent readers observed an incomplete atomic replacement: "
            f"{invalid_observations[0]}"
        )

    final_payload = json.loads(target.read_text(encoding="utf-8"))
    expected_pairs = {
        (worker_id, sequence)
        for worker_id in range(writers)
        for sequence in range(writes_per_worker)
    }
    final_pair = (final_payload.get("worker"), final_payload.get("sequence"))
    if final_pair not in expected_pairs:
        raise RuntimeError(f"unexpected final atomic replacement record: {final_payload!r}")

    temporary_files = sorted(path.name for path in directory.glob(".*.tmp"))
    if temporary_files:
        raise RuntimeError(f"atomic replacement leaked temporary files: {temporary_files}")

    return {
        "status": "passed",
        "writers": writers,
        "writes_per_worker": writes_per_worker,
        "reader_observations": observations,
        "reader_access_retries": reader_access_retries,
        "final_record": {"worker": final_pair[0], "sequence": final_pair[1]},
    }


def _run_jsonl_check(
    directory: Path,
    *,
    workers: int,
    records_per_worker: int,
) -> dict[str, Any]:
    target = directory / "metrics.jsonl"
    processes = [
        _launch_worker("jsonl", target, worker_id, records_per_worker)
        for worker_id in range(workers)
    ]
    _collect_workers(processes, "JSONL append")

    raw = target.read_bytes()
    if not raw.endswith(b"\n"):
        raise RuntimeError("concurrent JSONL output does not end at a record boundary")
    try:
        records = [json.loads(line) for line in raw.decode("utf-8").splitlines()]
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"concurrent JSONL output contains a partial record: {exc}") from exc

    expected = {
        (worker_id, sequence)
        for worker_id in range(workers)
        for sequence in range(records_per_worker)
    }
    observed = {(record.get("worker"), record.get("sequence")) for record in records}
    if observed != expected or len(records) != len(expected):
        raise RuntimeError(
            "concurrent JSONL records were lost or duplicated: "
            f"expected={len(expected)} unique={len(observed)} total={len(records)}"
        )

    return {
        "status": "passed",
        "workers": workers,
        "records_per_worker": records_per_worker,
        "records": len(records),
        "bytes": len(raw),
    }


def _synchronize_device(torch: Any, device_type: str) -> None:
    if device_type == "cuda":
        torch.cuda.synchronize()
    elif device_type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def _run_device_and_checkpoint_check(
    directory: Path,
    *,
    requested_device: str,
    required_device: str | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch

    from dlhub.checkpoint import load_checkpoint, save_checkpoint
    from dlhub.device import resolve_device

    resolved = resolve_device(requested_device)
    device = resolved.torch_device
    device_type = device.type
    if required_device is not None and device_type != required_device:
        raise RuntimeError(
            f"required device {required_device!r}, but {requested_device!r} resolved to "
            f"{device_type!r}"
        )

    torch.manual_seed(20260830)
    left = torch.arange(12, dtype=torch.float32, device=device).reshape(3, 4)
    right = torch.arange(8, dtype=torch.float32, device=device).reshape(4, 2)
    product = left @ right
    if product.device.type != device_type or not bool(torch.isfinite(product).all().item()):
        raise RuntimeError("tensor matmul did not execute successfully on the resolved device")

    model = torch.nn.Linear(4, 2).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.03, momentum=0.9)
    features = torch.tensor(
        [
            [-1.0, 0.5, 1.0, 0.0],
            [0.0, 1.0, -0.5, 2.0],
            [1.0, -1.0, 0.5, -0.5],
            [2.0, 0.0, -1.0, 1.0],
        ],
        device=device,
    )
    target = torch.stack(
        (features[:, 0] + 0.5 * features[:, 2], features[:, 1] - features[:, 3]),
        dim=1,
    )
    initial_parameters = [parameter.detach().clone() for parameter in model.parameters()]
    losses: list[float] = []
    for _ in range(12):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(model(features), target)
        if not bool(torch.isfinite(loss).item()):
            raise RuntimeError("training produced a non-finite loss")
        losses.append(float(loss.detach().cpu().item()))
        loss.backward()
        optimizer.step()
    _synchronize_device(torch, device_type)

    if losses[-1] >= losses[0]:
        raise RuntimeError(f"tiny training loss did not decrease: {losses[0]} -> {losses[-1]}")
    if not any(
        not torch.equal(before, after.detach())
        for before, after in zip(initial_parameters, model.parameters(), strict=True)
    ):
        raise RuntimeError("tiny training did not update any model parameter")

    expected_state = {
        name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()
    }
    checkpoint_path = directory / "runtime-checkpoint.pt"
    save_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        epoch=12,
        extra={"runtime_device": device_type},
    )

    restored = torch.nn.Linear(4, 2).cpu()
    restored_optimizer = torch.optim.SGD(restored.parameters(), lr=0.03, momentum=0.9)
    metadata = load_checkpoint(
        checkpoint_path,
        model=restored,
        optimizer=restored_optimizer,
        map_location="cpu",
    )
    for name, tensor in restored.state_dict().items():
        if tensor.device.type != "cpu":
            raise RuntimeError(f"checkpoint tensor {name!r} was not restored onto CPU")
        torch.testing.assert_close(tensor.detach(), expected_state[name])
    if metadata != {"epoch": 12, "extra": {"runtime_device": device_type}}:
        raise RuntimeError(f"checkpoint metadata changed during reload: {metadata!r}")
    optimizer_tensors = [
        value
        for state in restored_optimizer.state.values()
        for value in state.values()
        if isinstance(value, torch.Tensor)
    ]
    if not optimizer_tensors or any(value.device.type != "cpu" for value in optimizer_tensors):
        raise RuntimeError("optimizer state was not restored onto CPU")

    temporary_files = sorted(path.name for path in directory.glob(".*.tmp"))
    if temporary_files:
        raise RuntimeError(f"checkpoint write leaked temporary files: {temporary_files}")

    device_details: dict[str, Any] = {
        "requested": requested_device,
        "resolved": str(device),
        "type": device_type,
        "tensor_matmul": "passed",
        "training": "passed",
        "training_loss_start": losses[0],
        "training_loss_end": losses[-1],
    }
    if device_type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        device_details.update(
            {
                "cuda_build": torch.version.cuda,
                "cuda_device_name": properties.name,
                "cuda_capability": list(torch.cuda.get_device_capability(index)),
                "cuda_total_memory_bytes": properties.total_memory,
            }
        )
    elif device_type == "mps":
        mps_details: dict[str, Any] = {
            "mps_built": bool(torch.backends.mps.is_built()),
            "mps_available": bool(torch.backends.mps.is_available()),
            "mps_high_watermark_ratio": os.environ.get("PYTORCH_MPS_HIGH_WATERMARK_RATIO"),
        }
        for key, function_name in (
            ("mps_current_allocated_memory_bytes", "current_allocated_memory"),
            ("mps_driver_allocated_memory_bytes", "driver_allocated_memory"),
            ("mps_recommended_max_memory_bytes", "recommended_max_memory"),
        ):
            function = getattr(torch.mps, function_name, None)
            if function is not None:
                mps_details[key] = int(function())
        device_details.update(mps_details)

    checkpoint_details = {
        "status": "passed",
        "bytes": checkpoint_path.stat().st_size,
        "safe_loader": "weights_only=True",
        "serialized_from_device": device_type,
        "map_location": "cpu",
        "model_state": "matched",
        "optimizer_state": "loaded on cpu",
    }
    return device_details, checkpoint_details


def run_runtime_check(
    *,
    device: str = "auto",
    requirements: tuple[str, ...] = (),
    require_device: str | None = None,
    work_dir: str | Path | None = None,
    require_filesystem: str | None = None,
    atomic_writers: int = 3,
    atomic_writes: int = 4,
    jsonl_workers: int = 4,
    jsonl_records: int = 8,
    keep_workspace: bool = False,
    cleanup_timeout_seconds: float = 30.0,
) -> dict[str, Any]:
    """Run the platform checks and return machine-readable evidence."""

    for label, value in {
        "atomic_writers": atomic_writers,
        "atomic_writes": atomic_writes,
        "jsonl_workers": jsonl_workers,
        "jsonl_records": jsonl_records,
    }.items():
        if value < 1:
            raise ValueError(f"{label} must be at least 1")
    if cleanup_timeout_seconds <= 0:
        raise ValueError("cleanup_timeout_seconds must be greater than zero")

    base_directory = None if work_dir is None else Path(work_dir).expanduser().resolve()
    if base_directory is not None and not base_directory.is_dir():
        raise NotADirectoryError(f"work directory does not exist: {base_directory}")

    started = time.perf_counter()
    runtime_directory = Path(tempfile.mkdtemp(prefix="dlhub-platform-", dir=base_directory))
    try:
        filesystem = _filesystem_evidence(runtime_directory)
        effective_required_device = _resolve_requirements(
            requirements,
            legacy_device=require_device,
            legacy_filesystem=require_filesystem,
            filesystem=filesystem,
        )
        interrupted_write = _run_interrupted_write_check(runtime_directory)
        atomic_replace = _run_atomic_replace_check(
            runtime_directory,
            writers=atomic_writers,
            writes_per_worker=atomic_writes,
        )
        jsonl = _run_jsonl_check(
            runtime_directory,
            workers=jsonl_workers,
            records_per_worker=jsonl_records,
        )
        device_result, checkpoint = _run_device_and_checkpoint_check(
            runtime_directory,
            requested_device=device,
            required_device=effective_required_device,
        )
    finally:
        if keep_workspace:
            workspace_cleanup = {
                "status": "retained",
                "path": str(runtime_directory),
                "reason": "--keep-workspace requested external cleanup",
            }
        else:
            workspace_cleanup = _remove_runtime_directory(
                runtime_directory,
                cleanup_timeout_seconds,
            )

    import torch

    return {
        "schema_version": 1,
        "ok": True,
        "status": "passed",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": round(time.perf_counter() - started, 3),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
        "filesystem": filesystem,
        "device": device_result,
        "requirements": sorted(set(requirements)),
        "source": _source_evidence(),
        "ci": _ci_evidence(),
        "checks": {
            "interrupted_write": interrupted_write,
            "atomic_replace": atomic_replace,
            "concurrent_jsonl": jsonl,
            "checkpoint": checkpoint,
        },
        "workspace_cleanup": workspace_cleanup,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", help="device understood by dlhub.device")
    parser.add_argument(
        "--require",
        action="append",
        choices=("cuda", "mps", "windows", "nfs"),
        default=[],
        help="repeatable hard requirement backed by runtime evidence",
    )
    parser.add_argument(
        "--require-device",
        choices=("cpu", "cuda", "mps"),
        help="fail unless the resolved device has this real backend",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        help="existing directory on the storage boundary to exercise",
    )
    parser.add_argument(
        "--require-filesystem",
        help="fail unless mountinfo proves this filesystem type; 'nfs' accepts nfs or nfs4",
    )
    parser.add_argument("--atomic-writers", type=int, default=3)
    parser.add_argument("--atomic-writes", type=int, default=4)
    parser.add_argument("--jsonl-workers", type=int, default=4)
    parser.add_argument("--jsonl-records", type=int, default=8)
    parser.add_argument(
        "--keep-workspace",
        action="store_true",
        help="retain the generated workspace for an external mount harness to clean",
    )
    parser.add_argument(
        "--cleanup-timeout-seconds",
        type=float,
        default=30.0,
        help="fail explicitly if POSIX workspace removal exceeds this deadline",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="atomically write the structured success or failure result to this path",
    )
    parser.add_argument(
        "--_worker",
        choices=("atomic", "failing", "jsonl"),
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--_path", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--_worker-id", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--_iterations", type=int, help=argparse.SUPPRESS)
    return parser


def _write_output(path: Path, payload: dict[str, Any]) -> None:
    from dlhub.config import write_json

    write_json(path.expanduser().resolve(), payload)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args._worker is not None:
        if args._path is None or args._worker_id is None or args._iterations is None:
            raise SystemExit("internal worker arguments are incomplete")
        if args._worker == "atomic":
            _run_atomic_worker(args._path, args._worker_id, args._iterations)
        elif args._worker == "failing":
            _run_failing_writer(args._path)
        else:
            _run_jsonl_worker(args._path, args._worker_id, args._iterations)
        return 0

    try:
        result = run_runtime_check(
            device=args.device,
            requirements=tuple(args.require),
            require_device=args.require_device,
            work_dir=args.work_dir,
            require_filesystem=args.require_filesystem,
            atomic_writers=args.atomic_writers,
            atomic_writes=args.atomic_writes,
            jsonl_workers=args.jsonl_workers,
            jsonl_records=args.jsonl_records,
            keep_workspace=args.keep_workspace,
            cleanup_timeout_seconds=args.cleanup_timeout_seconds,
        )
        if not math.isfinite(float(result["duration_seconds"])):
            raise RuntimeError("invalid duration")
    except Exception as exc:
        result = {
            "schema_version": 1,
            "ok": False,
            "status": "failed",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "platform": {
                "system": platform.system(),
                "release": platform.release(),
                "machine": platform.machine(),
                "python": platform.python_version(),
            },
            "requirements": sorted(set(args.require)),
            "source": _source_evidence(),
            "ci": _ci_evidence(),
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }
        if args.output is not None:
            try:
                _write_output(args.output, result)
            except Exception as output_exc:
                print(
                    f"platform-runtime: could not write failure JSON: {output_exc}",
                    file=sys.stderr,
                )
        print(f"platform-runtime: FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    if args.output is not None:
        try:
            _write_output(args.output, result)
        except Exception as exc:
            print(f"platform-runtime: FAIL: could not write output JSON: {exc}", file=sys.stderr)
            return 1
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
