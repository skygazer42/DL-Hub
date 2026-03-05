"""Image denoising algorithms (toy-first, pure torch implementations).

Conventions:
- One algorithm family per file (variants live in that file).
- Each file exposes a `build_<name>_denoiser(...)` factory and a `__main__` smoke test.
"""

from .attention_unet import AttentionUNet, build_attention_unet_denoiser
from .aspp_unet import ASPPUNet, build_aspp_unet_denoiser
from .bm3d import BM3D, build_bm3d_denoiser
from .bsn import BlindSpotNet, build_bsn_denoiser
from .cbam_unet import CBAMUNet, build_cbam_unet_denoiser
from .cbdnet import CBDNet, build_cbdnet_denoiser
from .convnext_unet import ConvNeXtUNet, build_convnext_unet_denoiser
from .ddpm_unet import DDPMUNet, DDPMUNetDenoiser, build_ddpm_unet_denoiser
from .dhdn import DHDN, build_dhdn_denoiser
from .dbsn import DBSN, build_dbsn_denoiser
from .didn import DIDN, build_didn_denoiser
from .dncnn import DnCNN, DnCNNDenoiser, build_dncnn_denoiser
from .drrn import DRRN, build_drrn_denoiser
from .drunet import DRUNet, build_drunet_denoiser
from .edsr import EDSR, build_edsr_denoiser
from .ffdnet import FFDNetDenoiser, build_ffdnet_denoiser
from .gated_pixelcnn_bsn import GatedPixelCNNBSN, build_gated_pixelcnn_bsn_denoiser
from .hinet import HINet, build_hinet_denoiser
from .ircnn import IRCNN, build_ircnn_denoiser
from .memnet import MemNet, build_memnet_denoiser
from .mirnet import MIRNet, build_mirnet_denoiser
from .mprnet import MPRNet, build_mprnet_denoiser
from .mwcnn import MWCNN, build_mwcnn_denoiser
from .nafnet import NAFNet, build_nafnet_denoiser
from .nlrn import NLRN, build_nlrn_denoiser
from .noise2noise import Noise2NoiseUNet, build_noise2noise_denoiser
from .pixelcnn_bsn import PixelCNNBSN, build_pixelcnn_bsn_denoiser
from .pridnet import PRIDNet, build_pridnet_denoiser
from .rcan import RCAN, build_rcan_denoiser
from .rdn import RDN, build_rdn_denoiser
from .rednet import REDNet, build_rednet_denoiser
from .resunet import ResUNet, build_resunet_denoiser
from .restormer import Restormer, build_restormer_denoiser
from .ridnet import RIDNetDenoiser, build_ridnet_denoiser
from .scunet import SCUNet, build_scunet_denoiser
from .swinir import SwinIR, build_swinir_denoiser
from .unetpp import UNetPlusPlus, build_unetpp_denoiser
from .uformer import UFormer, build_uformer_denoiser

__all__ = [
    "AttentionUNet",
    "ASPPUNet",
    "BM3D",
    "BlindSpotNet",
    "CBAMUNet",
    "CBDNet",
    "ConvNeXtUNet",
    "DDPMUNet",
    "DDPMUNetDenoiser",
    "DHDN",
    "DBSN",
    "DIDN",
    "DnCNN",
    "DnCNNDenoiser",
    "DRRN",
    "DRUNet",
    "EDSR",
    "FFDNetDenoiser",
    "GatedPixelCNNBSN",
    "HINet",
    "IRCNN",
    "MemNet",
    "MIRNet",
    "MPRNet",
    "MWCNN",
    "NAFNet",
    "NLRN",
    "Noise2NoiseUNet",
    "PixelCNNBSN",
    "PRIDNet",
    "RCAN",
    "RDN",
    "REDNet",
    "ResUNet",
    "Restormer",
    "RIDNetDenoiser",
    "SCUNet",
    "SwinIR",
    "UNetPlusPlus",
    "UFormer",
    "build_attention_unet_denoiser",
    "build_aspp_unet_denoiser",
    "build_bm3d_denoiser",
    "build_bsn_denoiser",
    "build_cbam_unet_denoiser",
    "build_cbdnet_denoiser",
    "build_convnext_unet_denoiser",
    "build_ddpm_unet_denoiser",
    "build_dhdn_denoiser",
    "build_dbsn_denoiser",
    "build_didn_denoiser",
    "build_dncnn_denoiser",
    "build_drrn_denoiser",
    "build_drunet_denoiser",
    "build_edsr_denoiser",
    "build_ffdnet_denoiser",
    "build_gated_pixelcnn_bsn_denoiser",
    "build_hinet_denoiser",
    "build_ircnn_denoiser",
    "build_memnet_denoiser",
    "build_mirnet_denoiser",
    "build_mprnet_denoiser",
    "build_mwcnn_denoiser",
    "build_nafnet_denoiser",
    "build_nlrn_denoiser",
    "build_noise2noise_denoiser",
    "build_pixelcnn_bsn_denoiser",
    "build_pridnet_denoiser",
    "build_rcan_denoiser",
    "build_rdn_denoiser",
    "build_rednet_denoiser",
    "build_resunet_denoiser",
    "build_restormer_denoiser",
    "build_ridnet_denoiser",
    "build_scunet_denoiser",
    "build_swinir_denoiser",
    "build_unetpp_denoiser",
    "build_uformer_denoiser",
]
