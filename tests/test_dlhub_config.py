import json
from dataclasses import dataclass
from pathlib import Path

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
