import json
import os
from pathlib import Path

import pytest

from dlhub.config import append_jsonl


def test_append_jsonl_writes_one_json_object_per_line(tmp_path: Path) -> None:
    out = tmp_path / "metrics.jsonl"
    append_jsonl(out, {"epoch": 1, "loss": 1.0})
    append_jsonl(out, {"epoch": 2, "loss": 0.5})

    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0]) == {"epoch": 1, "loss": 1.0}
    assert json.loads(lines[1]) == {"epoch": 2, "loss": 0.5}


def test_append_jsonl_appends_one_complete_write_and_syncs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from dlhub import _atomic

    real_write = os.write
    real_fsync = os.fsync
    writes: list[bytes] = []
    synced: list[int] = []

    def track_write(file_descriptor: int, payload: bytes) -> int:
        writes.append(bytes(payload))
        return real_write(file_descriptor, payload)

    def track_fsync(file_descriptor: int) -> None:
        synced.append(file_descriptor)
        real_fsync(file_descriptor)

    monkeypatch.setattr(_atomic.os, "write", track_write)
    monkeypatch.setattr(_atomic.os, "fsync", track_fsync)

    out = tmp_path / "metrics.jsonl"
    append_jsonl(out, {"message": "你好", "step": 1})

    assert writes == ['{"message": "你好", "step": 1}\n'.encode()]
    assert synced
    assert out.read_bytes() == writes[0]
