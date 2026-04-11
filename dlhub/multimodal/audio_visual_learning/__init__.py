"""Audio-visual learning family package with lazy builder imports.

Conventions:
- One audio-visual family per file.
- Each family exposes `build_<family>_audio_visual_model(...)`.
- Each family keeps `_VARIANTS` and a `__main__` smoke test.
"""
from __future__ import annotations

from importlib import import_module
from typing import Any


_FAMILIES = (
    "av_syncnet",
    "av_contrast",
    "av_fusionnet",
    "av_localizer",
    "av_separation",
    "av_caption_bridge",
    "transformer_av",
    "diffusion_av",
    "prompt_av",
    "mamba_av",
)
_BUILDERS = [f"build_{family}_audio_visual_model" for family in _FAMILIES]


def _import_attr(name: str) -> Any:
    if name.startswith("build_") and name.endswith("_audio_visual_model"):
        stem = name[len("build_") : -len("_audio_visual_model")]
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


def __dir__() -> list[str]:
    return sorted({*globals().keys(), *_BUILDERS})


__all__ = list(_BUILDERS)
