import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from ._atomic import append_bytes, atomic_write


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
    """Atomically write ``payload`` as UTF-8 JSON.

    Serialization and durable staging complete before an existing destination
    is replaced, so failures cannot truncate the last good configuration.
    """

    out_path = Path(path)
    text = json.dumps(
        _to_jsonable(payload),
        ensure_ascii=False,
        indent=int(indent),
        sort_keys=True,
    )
    encoded = f"{text}\n".encode()

    def write(handle) -> None:
        handle.write(encoded)

    atomic_write(out_path, write)


def append_jsonl(path: str | Path, record: Any) -> None:
    """Append and sync one UTF-8 JSON record as a single write."""

    out_path = Path(path)
    line = json.dumps(_to_jsonable(record), ensure_ascii=False, sort_keys=True)
    append_bytes(out_path, f"{line}\n".encode())
