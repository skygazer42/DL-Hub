"""Text-to-3D family package with lazy builder imports.

Conventions:
- One text-to-3D family per file.
- Each family exposes `build_<family>_text3d_generator(...)`.
- Each family keeps `_VARIANTS` and a `__main__` smoke test.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


def _import_attr(name: str) -> Any:
    if name.startswith("build_") and name.endswith("_text3d_generator"):
        stem = name[len("build_") : -len("_text3d_generator")]
        module = import_module(f"{__name__}.{stem}")
        attr = getattr(module, name)
        globals()[name] = attr
        return attr
    raise AttributeError(name)


def __getattr__(name: str) -> Any:  # pragma: no cover
    try:
        return _import_attr(name)
    except AttributeError as exc:  # pragma: no cover
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc


def __dir__() -> list[str]:  # pragma: no cover
    return sorted(list(globals().keys()))


__all__ = []
