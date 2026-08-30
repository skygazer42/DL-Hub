from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import types

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "platform_runtime_check.py"
WORKFLOW_PATH = SCRIPT_PATH.parents[1] / ".github" / "workflows" / "platform-runtime.yml"


def _load_script() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("platform_runtime_check", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_native_platform_workflow_has_a_premerge_verification_trigger() -> None:
    workflow = WORKFLOW_PATH.read_text(encoding="utf-8")

    assert (
        """\
  push:
    branches:
      - 'platform-runtime-verify/**'
  workflow_dispatch:
"""
        in workflow
    )
    assert workflow.count("github.event_name == 'push'") == 2
    assert "runs-on: windows-2025" in workflow
    assert "runs-on: macos-15" in workflow
    assert "runs-on: macos-14" not in workflow
    assert "group: platform-runtime-${{ github.ref }}-${{ inputs.target || 'all' }}" in workflow
    # Pin the native matrix independently of the project's broad torch>=2.0
    # compatibility range so archived evidence remains reproducible.
    assert workflow.count('python -m pip install "torch==2.13.0"') == 2
    assert 'python -m pip install -e ".[torch]"' not in workflow
    assert "PYTORCH_MPS_HIGH_WATERMARK_RATIO" not in workflow
    assert "pull_request:" not in workflow

    action_refs = [
        line.split("@", 1)[1].split()[0]
        for line in workflow.splitlines()
        if line.strip().startswith("uses:")
    ]
    assert action_refs
    assert all(
        len(ref) == 40 and all(char in "0123456789abcdef" for char in ref) for ref in action_refs
    )


def test_linux_mountinfo_uses_longest_mount_and_decodes_paths() -> None:
    runtime_check = _load_script()
    mountinfo = """\
36 29 8:1 / / rw,relatime - ext4 /dev/sda rw
55 36 0:45 / /mnt/shared\\040data rw,relatime - nfs4 server:/export\\040data rw
"""

    result = runtime_check._linux_mount_for_path(
        mountinfo,
        Path("/mnt/shared data/runtime/check"),
    )

    assert result == {
        "device": "0:45",
        "type": "nfs4",
        "mount_point": "/mnt/shared data",
        "source": "server:/export data",
    }


def test_linux_mountinfo_prefers_newer_topmost_stacked_mount() -> None:
    runtime_check = _load_script()
    mountinfo = """\
8098 8087 8:1 /exports/client /client rw,relatime - ext4 /dev/sda rw
7773 8098 0:45 / /client rw,relatime - nfs4 127.0.0.1:/ rw,vers=4.2
"""

    result = runtime_check._linux_mount_for_path(
        mountinfo,
        Path("/client/runtime/check"),
        device_id=runtime_check._linux_device_id(0, 45),
    )

    assert result is not None
    assert result["type"] == "nfs4"
    assert result["source"] == "127.0.0.1:/"


def test_required_nfs_cannot_be_satisfied_by_a_local_filesystem() -> None:
    runtime_check = _load_script()

    with pytest.raises(RuntimeError, match="required filesystem 'nfs'.*'ext4'"):
        runtime_check._require_filesystem(
            {"type": "ext4", "evidence": "/proc/self/mountinfo"},
            "nfs",
        )

    runtime_check._require_filesystem(
        {"type": "nfs4", "evidence": "/proc/self/mountinfo"},
        "nfs",
    )


def test_cpu_runtime_gate_executes_and_removes_temporary_workspace(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    runtime_check = _load_script()

    result = runtime_check.run_runtime_check(
        device="cpu",
        require_device="cpu",
        work_dir=tmp_path,
        atomic_writers=2,
        atomic_writes=2,
        jsonl_workers=2,
        jsonl_records=3,
    )

    assert result["status"] == "passed"
    assert result["ok"] is True
    assert len(result["source"]["sha256"]["scripts/platform_runtime_check.py"]) == 64
    assert result["source"]["git_head"]
    assert result["device"]["type"] == "cpu"
    assert result["device"]["tensor_matmul"] == "passed"
    assert result["device"]["training"] == "passed"
    assert result["checks"]["atomic_replace"]["status"] == "passed"
    assert result["checks"]["interrupted_write"]["old_file"] == "preserved"
    assert result["checks"]["interrupted_write"]["temporary_files"] == "cleaned"
    assert result["checks"]["concurrent_jsonl"]["records"] == 6
    assert result["checks"]["checkpoint"]["safe_loader"] == "weights_only=True"
    assert result["checks"]["checkpoint"]["map_location"] == "cpu"
    assert result["checks"]["checkpoint"]["serialized_from_device"] == "cpu"
    if os.name == "nt":
        assert result["filesystem"]["type"] != "unverified"
        assert "GetVolumeInformationW" in result["filesystem"]["evidence"]
    assert result["workspace_cleanup"]["status"] == "removed"
    if os.name != "nt":
        assert result["workspace_cleanup"]["timeout_enforced"] is True
    assert list(tmp_path.iterdir()) == []


def test_atomic_replace_retries_transient_windows_sharing_violation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from dlhub import _atomic
    from dlhub.config import write_json

    target = tmp_path / "runtime-state.json"
    target.write_text('{"generation": "old"}\n', encoding="utf-8")
    real_replace = _atomic.os.replace
    attempts = 0

    def transient_replace(source: Path, destination: Path) -> None:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise PermissionError(5, "simulated Windows sharing violation")
        real_replace(source, destination)

    monkeypatch.setattr(_atomic, "_WINDOWS", True)
    monkeypatch.setattr(_atomic.os, "replace", transient_replace)

    write_json(target, {"generation": "new"})

    assert attempts == 3
    assert json.loads(target.read_text(encoding="utf-8")) == {"generation": "new"}
    assert list(tmp_path.iterdir()) == [target]


def test_windows_append_path_takes_and_releases_cross_process_lock(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from dlhub import _atomic

    class FakeMsvcrt:
        LK_LOCK = 1
        LK_UNLCK = 2

        def __init__(self) -> None:
            self.calls: list[tuple[int, int]] = []

        def locking(self, _file_descriptor: int, mode: int, bytes_to_lock: int) -> None:
            self.calls.append((mode, bytes_to_lock))

    fake_msvcrt = FakeMsvcrt()
    monkeypatch.setattr(_atomic, "_WINDOWS", True)
    monkeypatch.setattr(_atomic, "_msvcrt", fake_msvcrt)
    target = tmp_path / "metrics.jsonl"

    _atomic.append_bytes(target, b'{"step": 1}\n')

    assert target.read_bytes() == b'{"step": 1}\n'
    assert fake_msvcrt.calls == [(fake_msvcrt.LK_LOCK, 1), (fake_msvcrt.LK_UNLCK, 1)]


def test_atomic_reader_retries_only_transient_windows_permission_denied(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime_check = _load_script()
    target = tmp_path / "runtime-state.json"
    target.write_text('{"generation": "complete"}\n', encoding="utf-8")
    real_read_text = Path.read_text
    attempts = 0

    def transient_read_text(path: Path, *args, **kwargs) -> str:
        nonlocal attempts
        if path == target:
            attempts += 1
            if attempts < 3:
                raise PermissionError(13, "simulated Windows replacement window")
        return real_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", transient_read_text)

    payload, retries = runtime_check._read_atomic_json(
        target,
        retry_permission_denied=True,
    )

    assert payload == {"generation": "complete"}
    assert retries == 2

    attempts = 0
    with pytest.raises(PermissionError, match="simulated Windows replacement window"):
        runtime_check._read_atomic_json(target, retry_permission_denied=False)


def test_windows_requirement_fails_on_a_non_windows_host(tmp_path: Path) -> None:
    if platform.system() == "Windows":
        pytest.skip("non-Windows rejection contract")
    runtime_check = _load_script()

    with pytest.raises(RuntimeError, match="required platform 'windows'"):
        runtime_check.run_runtime_check(
            device="cpu",
            requirements=("windows",),
            work_dir=tmp_path,
            atomic_writers=1,
            atomic_writes=1,
            jsonl_workers=1,
            jsonl_records=1,
        )


def test_cli_writes_structured_failure_json(tmp_path: Path) -> None:
    output = tmp_path / "failure.json"

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--device",
            "cpu",
            "--require-filesystem",
            "definitely-not-a-real-filesystem",
            "--output",
            str(output),
        ],
        cwd=SCRIPT_PATH.parents[1],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 1
    assert "platform-runtime: FAIL" in completed.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["ok"] is False
    assert payload["status"] == "failed"
    assert payload["error"]["type"] == "RuntimeError"
    assert len(payload["source"]["sha256"]["dlhub/_atomic.py"]) == 64
