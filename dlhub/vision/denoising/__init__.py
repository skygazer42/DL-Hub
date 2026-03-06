"""Image denoising algorithms (toy-first, pure torch implementations).

Conventions:
- One algorithm family per file (variants live in that file).
- Each file exposes a `build_<name>_denoiser(...)` factory and a `__main__` smoke test.
"""

from .attention_unet import AttentionUNet, build_attention_unet_denoiser
from .anscombe_wiener import AnscombeWiener, build_anscombe_wiener_denoiser
from .anisotropic_diffusion import AnisotropicDiffusion, build_anisotropic_diffusion_denoiser
from .aspp_unet import ASPPUNet, build_aspp_unet_denoiser
from .bilateral_filter import BilateralFilter, build_bilateral_filter_denoiser
from .block_bias_corrector import BlockBiasCorrector, build_block_bias_corrector_denoiser
from .bm3d import BM3D, build_bm3d_denoiser
from .brdnet import BRDNet, build_brdnet_denoiser
from .bsn import BlindSpotNet, build_bsn_denoiser
from .carn import CARN, build_carn_denoiser
from .cbam_unet import CBAMUNet, build_cbam_unet_denoiser
from .cbdnet import CBDNet, build_cbdnet_denoiser
from .convnext_unet import ConvNeXtUNet, build_convnext_unet_denoiser
from .ddpm_unet import DDPMUNet, DDPMUNetDenoiser, build_ddpm_unet_denoiser
from .dead_hot_pixel_corrector import DeadHotPixelCorrector, build_dead_hot_pixel_corrector_denoiser
from .debanding_filter import DebandingFilter, build_debanding_filter_denoiser
from .dhdn import DHDN, build_dhdn_denoiser
from .dbsn import DBSN, build_dbsn_denoiser
from .denseunet import DenseUNet, build_denseunet_denoiser
from .didn import DIDN, build_didn_denoiser
from .dncnn import DnCNN, DnCNNDenoiser, build_dncnn_denoiser
from .drrn import DRRN, build_drrn_denoiser
from .drunet import DRUNet, build_drunet_denoiser
from .edsr import EDSR, build_edsr_denoiser
from .ffdnet import FFDNetDenoiser, build_ffdnet_denoiser
from .gated_pixelcnn_bsn import GatedPixelCNNBSN, build_gated_pixelcnn_bsn_denoiser
from .guided_filter import GuidedFilter, build_guided_filter_denoiser
from .hinet import HINet, build_hinet_denoiser
from .ircnn import IRCNN, build_ircnn_denoiser
from .kuan_filter import KuanFilter, build_kuan_filter_denoiser
from .lee_filter import LeeFilter, build_lee_filter_denoiser
from .line_defect_corrector import LineDefectCorrector, build_line_defect_corrector_denoiser
from .median_filter import MedianFilter, build_median_filter_denoiser
from .memnet import MemNet, build_memnet_denoiser
from .mirnet import MIRNet, build_mirnet_denoiser
from .mprnet import MPRNet, build_mprnet_denoiser
from .mwcnn import MWCNN, build_mwcnn_denoiser
from .nafnet import NAFNet, build_nafnet_denoiser
from .nlrn import NLRN, build_nlrn_denoiser
from .non_local_means import NonLocalMeans, build_non_local_means_denoiser
from .noise2noise import Noise2NoiseUNet, build_noise2noise_denoiser
from .pixelcnn_bsn import PixelCNNBSN, build_pixelcnn_bsn_denoiser
from .pridnet import PRIDNet, build_pridnet_denoiser
from .r2unet import R2UNet, build_r2unet_denoiser
from .rcan import RCAN, build_rcan_denoiser
from .rdn import RDN, build_rdn_denoiser
from .rednet import REDNet, build_rednet_denoiser
from .resunet import ResUNet, build_resunet_denoiser
from .restormer import Restormer, build_restormer_denoiser
from .ridnet import RIDNetDenoiser, build_ridnet_denoiser
from .rrdbnet import RRDBNet, build_rrdbnet_denoiser
from .scunet import SCUNet, build_scunet_denoiser
from .rowcol_bias_corrector import RowColBiasCorrector, build_rowcol_bias_corrector_denoiser
from .swinir import SwinIR, build_swinir_denoiser
from .stripe_remover import StripeRemover, build_stripe_remover_denoiser
from .total_variation import TotalVariationDenoiser, build_total_variation_denoiser
from .unet3plus import UNet3Plus, build_unet3plus_denoiser
from .unetpp import UNetPlusPlus, build_unetpp_denoiser
from .uformer import UFormer, build_uformer_denoiser
from .wavelet_shrinkage import WaveletShrinkage, build_wavelet_shrinkage_denoiser
from .wiener_filter import WienerFilter, build_wiener_filter_denoiser

__all__ = [
    "AttentionUNet",
    "AnscombeWiener",
    "AnisotropicDiffusion",
    "ASPPUNet",
    "BilateralFilter",
    "BlockBiasCorrector",
    "BM3D",
    "BRDNet",
    "BlindSpotNet",
    "CARN",
    "CBAMUNet",
    "CBDNet",
    "ConvNeXtUNet",
    "DDPMUNet",
    "DDPMUNetDenoiser",
    "DeadHotPixelCorrector",
    "DebandingFilter",
    "DHDN",
    "DBSN",
    "DenseUNet",
    "DIDN",
    "DnCNN",
    "DnCNNDenoiser",
    "DRRN",
    "DRUNet",
    "EDSR",
    "FFDNetDenoiser",
    "GatedPixelCNNBSN",
    "GuidedFilter",
    "HINet",
    "IRCNN",
    "KuanFilter",
    "LeeFilter",
    "LineDefectCorrector",
    "MedianFilter",
    "MemNet",
    "MIRNet",
    "MPRNet",
    "MWCNN",
    "NAFNet",
    "NLRN",
    "NonLocalMeans",
    "Noise2NoiseUNet",
    "PixelCNNBSN",
    "PRIDNet",
    "R2UNet",
    "RCAN",
    "RDN",
    "REDNet",
    "ResUNet",
    "Restormer",
    "RIDNetDenoiser",
    "RRDBNet",
    "SCUNet",
    "RowColBiasCorrector",
    "SwinIR",
    "StripeRemover",
    "TotalVariationDenoiser",
    "UNet3Plus",
    "UNetPlusPlus",
    "UFormer",
    "WaveletShrinkage",
    "WienerFilter",
    "build_attention_unet_denoiser",
    "build_anscombe_wiener_denoiser",
    "build_anisotropic_diffusion_denoiser",
    "build_aspp_unet_denoiser",
    "build_bilateral_filter_denoiser",
    "build_block_bias_corrector_denoiser",
    "build_bm3d_denoiser",
    "build_brdnet_denoiser",
    "build_bsn_denoiser",
    "build_carn_denoiser",
    "build_cbam_unet_denoiser",
    "build_cbdnet_denoiser",
    "build_convnext_unet_denoiser",
    "build_ddpm_unet_denoiser",
    "build_dead_hot_pixel_corrector_denoiser",
    "build_debanding_filter_denoiser",
    "build_dhdn_denoiser",
    "build_dbsn_denoiser",
    "build_denseunet_denoiser",
    "build_didn_denoiser",
    "build_dncnn_denoiser",
    "build_drrn_denoiser",
    "build_drunet_denoiser",
    "build_edsr_denoiser",
    "build_ffdnet_denoiser",
    "build_gated_pixelcnn_bsn_denoiser",
    "build_guided_filter_denoiser",
    "build_hinet_denoiser",
    "build_ircnn_denoiser",
    "build_kuan_filter_denoiser",
    "build_lee_filter_denoiser",
    "build_line_defect_corrector_denoiser",
    "build_median_filter_denoiser",
    "build_memnet_denoiser",
    "build_mirnet_denoiser",
    "build_mprnet_denoiser",
    "build_mwcnn_denoiser",
    "build_nafnet_denoiser",
    "build_nlrn_denoiser",
    "build_non_local_means_denoiser",
    "build_noise2noise_denoiser",
    "build_pixelcnn_bsn_denoiser",
    "build_pridnet_denoiser",
    "build_r2unet_denoiser",
    "build_rcan_denoiser",
    "build_rdn_denoiser",
    "build_rednet_denoiser",
    "build_resunet_denoiser",
    "build_restormer_denoiser",
    "build_ridnet_denoiser",
    "build_rrdbnet_denoiser",
    "build_scunet_denoiser",
    "build_rowcol_bias_corrector_denoiser",
    "build_swinir_denoiser",
    "build_stripe_remover_denoiser",
    "build_total_variation_denoiser",
    "build_unet3plus_denoiser",
    "build_unetpp_denoiser",
    "build_uformer_denoiser",
    "build_wavelet_shrinkage_denoiser",
    "build_wiener_filter_denoiser",
]
