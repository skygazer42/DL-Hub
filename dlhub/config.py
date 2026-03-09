
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


def _to_jsonable(obj: Any) -> Any:
    if is_dataclass(obj):
        return {k: _to_jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_to_jsonable(v) for v in obj]
    return obj


def dataclass_to_dict(obj: Any) -> dict[str, Any]:
    """Convert a (possibly nested) dataclass into a JSON-serializable dict."""

    converted = _to_jsonable(obj)
    if not isinstance(converted, dict):
        raise TypeError(f"Expected dataclass to convert into dict, got {type(converted).__name__}")
    return converted


def write_json(path: str | Path, payload: Any, *, indent: int = 2) -> None:
    """Write `payload` as JSON (UTF-8), creating parent directories if needed."""

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(_to_jsonable(payload), f, ensure_ascii=False, indent=int(indent), sort_keys=True)
        f.write("\n")


def append_jsonl(path: str | Path, record: Any) -> None:
    """Append one JSON record as a single line (JSONL)."""

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(_to_jsonable(record), ensure_ascii=False, sort_keys=True)
    with out_path.open("a", encoding="utf-8") as f:
        f.write(line)
        f.write("\n")
