"""Semantic segmentation models (pure torch, toy-first).

Conventions:
- One algorithm family per file (variants live in that file).
- Each file exposes a `build_<name>_segmenter(...)` factory and a `__main__` smoke test.
"""

from .unet import UNetSegmenter, build_unet_segmenter
from .deeplabv3plus import DeepLabV3Plus, build_deeplabv3plus_segmenter
from .pspnet import PSPNet, build_pspnet_segmenter

__all__ = [
    "DeepLabV3Plus",
    "PSPNet",
    "UNetSegmenter",
    "build_deeplabv3plus_segmenter",
    "build_pspnet_segmenter",
    "build_unet_segmenter",
]

