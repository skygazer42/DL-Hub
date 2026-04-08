"""Medical segmentation models (toy-first, pure torch)."""
from __future__ import annotations
from importlib import import_module
from typing import Any
def _import_attr(name: str) -> Any:
    if name.startswith("build_") and name.endswith("_medical_segmenter"):
        stem = name[len("build_") : -len("_medical_segmenter")]
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
__all__: list[str] = []
