"""Image denoising algorithms (toy-first, pure torch implementations).

Conventions:
- One algorithm family per file (variants live in that file).
- Each file exposes a `build_<name>_denoiser(...)` factory and a `__main__` smoke test.
"""

from .bm3d import BM3D, build_bm3d_denoiser
from .dncnn import DnCNN, DnCNNDenoiser, build_dncnn_denoiser
from .noise2noise import Noise2NoiseUNet, build_noise2noise_denoiser
from .restormer import Restormer, build_restormer_denoiser

__all__ = [
    "BM3D",
    "DnCNN",
    "DnCNNDenoiser",
    "Noise2NoiseUNet",
    "Restormer",
    "build_bm3d_denoiser",
    "build_dncnn_denoiser",
    "build_noise2noise_denoiser",
    "build_restormer_denoiser",
]

