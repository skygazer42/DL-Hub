"""Instance segmentation models (pure torch, toy-first).

Conventions:
- One algorithm family per file (variants live in that file).
- Each file exposes a `build_<name>_instance_segmenter(...)` factory and a `__main__` smoke test.
"""

from .yolact import YOLACTLite, build_yolact_instance_segmenter

__all__ = [
    "YOLACTLite",
    "build_yolact_instance_segmenter",
]

