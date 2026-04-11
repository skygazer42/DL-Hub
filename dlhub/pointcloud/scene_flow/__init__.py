"""Point cloud scene flow estimators (pure torch, toy-first).

Conventions:
- One scene-flow family per file (variants live in that file via `_VARIANTS`).
- Each family exposes a `build_<name>_scene_flow_estimator(...)` factory and a `__main__` smoke test.

This package uses lazy imports so `import dlhub.pointcloud.scene_flow` stays lightweight.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


def _import_attr(name: str) -> Any:
    if name.startswith("build_") and name.endswith("_scene_flow_estimator"):
        stem = name[len("build_") : -len("_scene_flow_estimator")]
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


__all__: list[str] = []
