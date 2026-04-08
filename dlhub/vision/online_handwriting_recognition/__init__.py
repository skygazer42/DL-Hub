"""Online handwriting recognition models (toy-first, pure torch)."""
from __future__ import annotations
from importlib import import_module
from typing import Any
def _import_attr(name: str) -> Any:
    if name.startswith("build_") and name.endswith("_handwriting_recognizer"):
        stem = name[len("build_") : -len("_handwriting_recognizer")]
        module = import_module(f"{__name__}.{stem}")
        attr = getattr(module, name)
        globals()[name] = attr
        return attr
    raise AttributeError(name)
def __getattr__(name: str) -> Any:
    try:
        return _import_attr(name)
    except AttributeError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
def __dir__() -> list[str]: return sorted(list(globals().keys()))
__all__: list[str] = []
