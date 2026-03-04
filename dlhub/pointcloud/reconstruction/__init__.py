"""Point cloud reconstruction models (pure torch, toy-first).

Conventions:
- One algorithm family per file (variants live in that file).
- Each file exposes a `build_<name>_...(...)` factory and a `__main__` smoke test.
"""

from .pointnet_ae import PointNetAutoEncoder, build_pointnet_autoencoder

__all__ = [
    "PointNetAutoEncoder",
    "build_pointnet_autoencoder",
]

