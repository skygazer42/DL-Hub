import json
from pathlib import Path

from dlhub.config import append_jsonl


def test_append_jsonl_writes_one_json_object_per_line(tmp_path: Path) -> None:
    out = tmp_path / "metrics.jsonl"
    append_jsonl(out, {"epoch": 1, "loss": 1.0})
    append_jsonl(out, {"epoch": 2, "loss": 0.5})

    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0]) == {"epoch": 1, "loss": 1.0}
    assert json.loads(lines[1]) == {"epoch": 2, "loss": 0.5}
