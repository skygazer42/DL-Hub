"""Small presentation helpers shared by repository command-line tools."""

from __future__ import annotations

from collections.abc import Iterable


def summarize_output(value: object) -> str:
    """Describe nested model outputs without printing full tensor contents."""

    try:
        import torch
    except (ImportError, OSError, RuntimeError):
        torch = None

    if torch is not None and isinstance(value, torch.Tensor):
        return f"Tensor(shape={tuple(value.shape)}, dtype={value.dtype}, device={value.device})"
    if isinstance(value, dict):
        keys = ", ".join(sorted(map(str, value.keys())))
        return f"dict(keys=[{keys}])"
    if isinstance(value, list | tuple):
        head = ", ".join(summarize_output(item) for item in value[:2])
        tail = "" if len(value) <= 2 else f", ... (+{len(value) - 2})"
        return f"{type(value).__name__}([{head}{tail}])"
    if hasattr(value, "logits"):
        return f"{type(value).__name__}(logits={summarize_output(getattr(value, 'logits'))})"
    return type(value).__name__


def print_limited(lines: Iterable[str], *, limit: int = 80, tail: int = 10) -> None:
    """Print a bounded list while retaining useful entries from both ends."""

    rows = list(lines)
    limit = int(limit)
    tail = int(tail)
    if limit <= 0:
        return
    if tail < 0:
        raise ValueError("tail must be >= 0")
    if len(rows) <= limit:
        for row in rows:
            print(row)
        return

    tail = min(tail, limit)
    head = limit - tail
    for row in rows[:head]:
        print(row)
    print(f"... ({len(rows) - limit} more) ...")
    if tail:
        for row in rows[-tail:]:
            print(row)


__all__ = ["print_limited", "summarize_output"]
