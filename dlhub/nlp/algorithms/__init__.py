"""NLP local zoo algorithm families (torch-only).

This package groups many architecture variants under a single algorithm-family module
(e.g. `transformer.py` contains the full Transformer variant registry), similar to
ResNet-style variants in vision.
"""

from .registry import REGISTRY, build_registry

__all__ = ["REGISTRY", "build_registry"]
