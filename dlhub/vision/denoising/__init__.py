"""Image denoising algorithms (toy-first, pure torch implementations).

Conventions:
- One algorithm family per file (variants live in that file).
- Each file exposes a `build_<name>_denoiser(...)` factory and a `__main__` smoke test.
"""

from .bm3d import BM3D, build_bm3d_denoiser
from .ddpm_unet import DDPMUNet, DDPMUNetDenoiser, build_ddpm_unet_denoiser
from .dncnn import DnCNN, DnCNNDenoiser, build_dncnn_denoiser
from .drunet import DRUNet, build_drunet_denoiser
from .ffdnet import FFDNetDenoiser, build_ffdnet_denoiser
from .nafnet import NAFNet, build_nafnet_denoiser
from .noise2noise import Noise2NoiseUNet, build_noise2noise_denoiser
from .restormer import Restormer, build_restormer_denoiser
from .ridnet import RIDNetDenoiser, build_ridnet_denoiser
from .swinir import SwinIR, build_swinir_denoiser

__all__ = [
    "BM3D",
    "DDPMUNet",
    "DDPMUNetDenoiser",
    "DnCNN",
    "DnCNNDenoiser",
    "DRUNet",
    "FFDNetDenoiser",
    "NAFNet",
    "Noise2NoiseUNet",
    "Restormer",
    "RIDNetDenoiser",
    "SwinIR",
    "build_bm3d_denoiser",
    "build_ddpm_unet_denoiser",
    "build_dncnn_denoiser",
    "build_drunet_denoiser",
    "build_ffdnet_denoiser",
    "build_nafnet_denoiser",
    "build_noise2noise_denoiser",
    "build_restormer_denoiser",
    "build_ridnet_denoiser",
    "build_swinir_denoiser",
]
