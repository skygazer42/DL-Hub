import json
from dataclasses import dataclass
import os
from pathlib import Path
import stat

import pytest

from dlhub.config import dataclass_to_dict, write_json


@dataclass(frozen=True)
class _DemoConfig:
    epochs: int
    learning_rate: float
    run_dir: Path


def test_write_json_round_trip(tmp_path: Path) -> None:
    cfg = _DemoConfig(epochs=1, learning_rate=1e-3, run_dir=tmp_path / "run")
    payload = dataclass_to_dict(cfg)

    out = tmp_path / "config.json"
    write_json(out, payload)

    text = out.read_text(encoding="utf-8")
    assert json.loads(text) == payload
    assert '"epochs": 1' in text
    assert '"learning_rate": 0.001' in text


def test_write_json_serialization_failure_preserves_existing_file(tmp_path: Path) -> None:
    out = tmp_path / "config.json"
    original = b'{"status": "last-good"}\n'
    out.write_bytes(original)

    with pytest.raises(TypeError):
        write_json(out, {"unsupported": object()})

    assert out.read_bytes() == original
    assert list(tmp_path.iterdir()) == [out]


def test_write_json_replace_failure_preserves_existing_file_and_cleans_temp(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from dlhub import _atomic

    out = tmp_path / "config.json"
    original = b'{"status": "last-good"}\n'
    out.write_bytes(original)

    def fail_replace(source, destination) -> None:
        del source, destination
        raise OSError("simulated replace failure")

    monkeypatch.setattr(_atomic.os, "replace", fail_replace)

    with pytest.raises(OSError, match="simulated replace failure"):
        write_json(out, {"status": "new"})

    assert out.read_bytes() == original
    assert list(tmp_path.iterdir()) == [out]


@pytest.mark.skipif(os.name == "nt", reason="POSIX mode bits and umask semantics")
def test_write_json_new_file_honors_process_umask(tmp_path: Path) -> None:
    out = tmp_path / "config.json"
    previous_umask = os.umask(0o027)
    try:
        write_json(out, {"status": "new"})
    finally:
        os.umask(previous_umask)

    assert stat.S_IMODE(out.stat().st_mode) == 0o640


@pytest.mark.skipif(os.name == "nt", reason="POSIX mode bits")
def test_write_json_replacement_preserves_existing_mode(tmp_path: Path) -> None:
    out = tmp_path / "config.json"
    out.write_text('{"status": "old"}\n', encoding="utf-8")
    out.chmod(0o664)

    write_json(out, {"status": "new"})

    assert stat.S_IMODE(out.stat().st_mode) == 0o664
