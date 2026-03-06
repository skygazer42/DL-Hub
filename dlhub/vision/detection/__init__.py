"""Object detection models (pure torch, toy-first).

Conventions:
- One algorithm family per file (variants live in that file).
- Each algorithm file exposes a `build_<name>_detector(...)` factory and a `__main__` smoke test.

This package uses **lazy imports** so `import dlhub.vision.detection` stays lightweight even as the zoo grows.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


# --- Lazy import routing
#
# Convention:
#   builder `build_<stem>_detector` is implemented in `dlhub/vision/detection/<stem>.py`
#
# Exceptions are listed here.
_STEM_TO_MODULE: dict[str, str] = {
    # Historical file name: YOLOv1 lives in yolo.py
    "yolo_v1": "yolo",
}


def _import_attr(name: str) -> Any:
    if name.startswith("build_") and name.endswith("_detector"):
        stem = name[len("build_") : -len("_detector")]
        module_name = _STEM_TO_MODULE.get(stem, stem)
        module = import_module(f"{__name__}.{module_name}")
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


__all__ = [
    # Builders are exposed via __getattr__ (lazy), but listing common ones here improves discoverability.
    "build_centernet_detector",
    "build_dino_detector",
    "build_dssd_detector",
    "build_efficientdet_detector",
    "build_fcos_detector",
    "build_retinanet_detector",
    "build_ssd_detector",
    "build_squeezedet_detector",
    "build_yolo_v1_detector",
    "build_yolov8_detector",
]
