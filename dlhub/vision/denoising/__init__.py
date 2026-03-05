"""Image denoising algorithms (toy-first, pure torch implementations).

Conventions:
- One algorithm family per file (variants live in that file).
- Each file exposes a `build_<name>_denoiser(...)` factory and a `__main__` smoke test.
"""

from .bm3d import BM3D, build_bm3d_denoiser
from .bsn import BlindSpotNet, build_bsn_denoiser
from .cbdnet import CBDNet, build_cbdnet_denoiser
from .ddpm_unet import DDPMUNet, DDPMUNetDenoiser, build_ddpm_unet_denoiser
from .didn import DIDN, build_didn_denoiser
from .dncnn import DnCNN, DnCNNDenoiser, build_dncnn_denoiser
from .drunet import DRUNet, build_drunet_denoiser
from .ffdnet import FFDNetDenoiser, build_ffdnet_denoiser
from .mirnet import MIRNet, build_mirnet_denoiser
from .mprnet import MPRNet, build_mprnet_denoiser
from .nafnet import NAFNet, build_nafnet_denoiser
from .noise2noise import Noise2NoiseUNet, build_noise2noise_denoiser
from .pixelcnn_bsn import PixelCNNBSN, build_pixelcnn_bsn_denoiser
from .rcan import RCAN, build_rcan_denoiser
from .restormer import Restormer, build_restormer_denoiser
from .ridnet import RIDNetDenoiser, build_ridnet_denoiser
from .swinir import SwinIR, build_swinir_denoiser
from .uformer import UFormer, build_uformer_denoiser

__all__ = [
    "BM3D",
    "BlindSpotNet",
    "CBDNet",
    "DDPMUNet",
    "DDPMUNetDenoiser",
    "DIDN",
    "DnCNN",
    "DnCNNDenoiser",
    "DRUNet",
    "FFDNetDenoiser",
    "MIRNet",
    "MPRNet",
    "NAFNet",
    "Noise2NoiseUNet",
    "PixelCNNBSN",
    "RCAN",
    "Restormer",
    "RIDNetDenoiser",
    "SwinIR",
    "UFormer",
    "build_bm3d_denoiser",
    "build_bsn_denoiser",
    "build_cbdnet_denoiser",
    "build_ddpm_unet_denoiser",
    "build_didn_denoiser",
    "build_dncnn_denoiser",
    "build_drunet_denoiser",
    "build_ffdnet_denoiser",
    "build_mirnet_denoiser",
    "build_mprnet_denoiser",
    "build_nafnet_denoiser",
    "build_noise2noise_denoiser",
    "build_pixelcnn_bsn_denoiser",
    "build_rcan_denoiser",
    "build_restormer_denoiser",
    "build_ridnet_denoiser",
    "build_swinir_denoiser",
    "build_uformer_denoiser",
]
