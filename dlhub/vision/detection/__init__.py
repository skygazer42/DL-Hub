"""Object detection models (pure torch, toy-first).

Conventions:
- One algorithm family per file (variants live in that file).
- Each file exposes a `build_<name>_detector(...)` factory and a `__main__` smoke test.
"""

from .fcos import FCOSDetector, build_fcos_detector
from .retinanet import RetinaNetDetector, build_retinanet_detector
from .centernet import CenterNetDetector, build_centernet_detector

__all__ = [
    "CenterNetDetector",
    "FCOSDetector",
    "RetinaNetDetector",
    "build_centernet_detector",
    "build_fcos_detector",
    "build_retinanet_detector",
]

