"""3D object detection models (pure torch, toy-first).

Conventions:
- One algorithm family per file (variants live in that file).
- Each algorithm file exposes a `build_<name>_detector3d(...)` factory and a `__main__` smoke test.

This package uses **lazy imports** so `import dlhub.pointcloud.detection3d` stays lightweight
even as the zoo grows.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


def _import_attr(name: str) -> Any:
    if name.startswith("build_") and name.endswith("_detector3d"):
        stem = name[len("build_") : -len("_detector3d")]
        module = import_module(f"{__name__}.{stem}")
        attr = getattr(module, name)
        globals()[name] = attr  # cache for next lookup
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

