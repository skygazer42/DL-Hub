"""Federated learning strategies (toy-first, no networking).

Conventions:
- One federated algorithm family per file via `_VARIANTS`.
- Each file exposes a `build_<name>_strategy(...)` factory.
- Strategies are simulation-oriented and implement `simulate_round(...)`.

This package uses lazy imports so `import dlhub.federated` stays lightweight.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


def _import_attr(name: str) -> Any:
    if name.startswith("build_") and name.endswith("_strategy"):
        stem = name[len("build_") : -len("_strategy")]
        module = import_module(f"{__name__}.{stem}")
        attr = getattr(module, name)
        globals()[name] = attr
        return attr
    raise AttributeError(name)


def __getattr__(name: str) -> Any:  # pragma: no cover
    try:
        return _import_attr(name)
    except AttributeError as e:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from e


def __dir__() -> list[str]:  # pragma: no cover
    return sorted(list(globals().keys()))


__all__ = []
