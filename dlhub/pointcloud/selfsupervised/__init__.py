"""Self-supervised learning for point clouds (pure torch, compact-first)."""

from .barlowtwins import BarlowTwinsPointNet, barlow_twins_loss, build_barlowtwins_pointnet
from .barlow2 import Barlow2PointNet, barlow_twins_loss as barlow2_loss, build_barlow2_pointnet
from .byol import BYOLPointNet, build_byol_pointnet, byol_loss, cosine_similarity_loss
from .byol2 import BYOL2PointNet, build_byol2_pointnet, byol2_loss
from .byol3_point import BYOL3PointNet, build_byol3_point_pointnet, byol3_point_loss
from .bootstrap_point import BootstrapPointNet, build_bootstrap_point_pointnet
from .clip_point_ssl import ClipPointSSLPretrainer, build_clip_point_ssl_pretrainer
from .cluster_ssl_point import ClusterSSLPointPretrainer, build_cluster_ssl_point_pretrainer
from .data2vec import Data2VecPointMAE, build_data2vec_pointmae, data2vec_loss
from .dino import DINOPointNet, build_dino_pointnet, dino_loss
from .dino2_point import DINO2PointNet, build_dino2_point_pointnet, dino2_point_loss
from .dinossl_point import DinoSSLPointPretrainer, build_dinossl_point_pretrainer
from .dinov2 import DINOV2PointMAE, build_dinov2_pointmae, dino_cross_view_loss, ibot_patch_loss
from .ibot_point import IBotPointPretrainer, build_ibot_point_pretrainer
from .ijepa import IJEPAPointMAE, build_ijepa_pointmae, ijepa_patch_loss
from .ijepa2_point import IJEPA2PointMAE, build_ijepa2_point_pointmae, ijepa2_point_patch_loss
from .jepa_point import JepaPointPretrainer, build_jepa_point_pretrainer
from .mae3d_point import Mae3DPointPretrainer, build_mae3d_point_pretrainer
from .maessl_point import MaeSSLPointPretrainer, build_maessl_point_pretrainer
from .maskedpoint import MaskedpointPretrainer, build_maskedpoint_pretrainer
from .moco import MoCoPointNet, build_moco_pointnet, moco_logits
from .mocov3_point import MoCoV3PointNet, build_mocov3_point_pointnet, mocov3_point_logits
from .msn import MSNPointMAE, build_msn_pointmae, msn_loss
from .pointmae import PointMAEPretrainer, build_pointmae_pretrainer
from .ressl import ReSSLPointNet, build_ressl_pointnet, ressl_loss
from .simclr import SimCLRPointNet, build_simclr_pointnet, nt_xent_loss
from .simclr2 import SimCLR2PointNet, build_simclr2_pointnet
from .simclrv3_point import SimCLRV3PointNet, build_simclrv3_point_pointnet
from .simmim_point import SimMimPointPretrainer, build_simmim_point_pretrainer
from .simsiam import (
    SimSiamPointNet,
    build_simsiam_pointnet,
    negative_cosine_similarity,
    simsiam_loss,
)
from .swav import SwAVPointNet, build_swav_pointnet, sinkhorn_knopp, swav_loss
from .swav2 import SwAV2PointNet, build_swav2_pointnet, swav2_loss
from .vicreg import VICRegPointNet, build_vicreg_pointnet, vicreg_loss
from .vicreg2 import VICReg2PointNet, build_vicreg2_pointnet, vicreg2_loss

__all__ = [
    "Barlow2PointNet",
    "BarlowTwinsPointNet",
    "BootstrapPointNet",
    "BYOLPointNet",
    "BYOL2PointNet",
    "BYOL3PointNet",
    "ClipPointSSLPretrainer",
    "ClusterSSLPointPretrainer",
    "Data2VecPointMAE",
    "DINOPointNet",
    "DINO2PointNet",
    "DinoSSLPointPretrainer",
    "DINOV2PointMAE",
    "IBotPointPretrainer",
    "IJEPA2PointMAE",
    "IJEPAPointMAE",
    "JepaPointPretrainer",
    "Mae3DPointPretrainer",
    "MaeSSLPointPretrainer",
    "MaskedpointPretrainer",
    "MSNPointMAE",
    "MoCoPointNet",
    "MoCoV3PointNet",
    "PointMAEPretrainer",
    "ReSSLPointNet",
    "SimCLRPointNet",
    "SimCLR2PointNet",
    "SimCLRV3PointNet",
    "SimMimPointPretrainer",
    "SimSiamPointNet",
    "SwAVPointNet",
    "SwAV2PointNet",
    "VICRegPointNet",
    "VICReg2PointNet",
    "barlow_twins_loss",
    "build_data2vec_pointmae",
    "build_dinov2_pointmae",
    "build_barlow2_pointnet",
    "build_barlowtwins_pointnet",
    "build_byol_pointnet",
    "build_byol2_pointnet",
    "build_byol3_point_pointnet",
    "build_bootstrap_point_pointnet",
    "build_clip_point_ssl_pretrainer",
    "build_cluster_ssl_point_pretrainer",
    "build_dino_pointnet",
    "build_dino2_point_pointnet",
    "build_dinossl_point_pretrainer",
    "build_ibot_point_pretrainer",
    "build_ijepa_pointmae",
    "build_ijepa2_point_pointmae",
    "build_jepa_point_pretrainer",
    "build_mae3d_point_pretrainer",
    "build_maessl_point_pretrainer",
    "build_maskedpoint_pretrainer",
    "build_msn_pointmae",
    "build_moco_pointnet",
    "build_mocov3_point_pointnet",
    "build_pointmae_pretrainer",
    "build_ressl_pointnet",
    "build_simclr_pointnet",
    "build_simclr2_pointnet",
    "build_simclrv3_point_pointnet",
    "build_simmim_point_pretrainer",
    "build_simsiam_pointnet",
    "build_swav_pointnet",
    "build_swav2_pointnet",
    "build_vicreg_pointnet",
    "build_vicreg2_pointnet",
    "barlow2_loss",
    "byol_loss",
    "byol2_loss",
    "byol3_point_loss",
    "cosine_similarity_loss",
    "data2vec_loss",
    "dino_cross_view_loss",
    "dino_loss",
    "dino2_point_loss",
    "ibot_patch_loss",
    "ijepa_patch_loss",
    "ijepa2_point_patch_loss",
    "msn_loss",
    "moco_logits",
    "mocov3_point_logits",
    "negative_cosine_similarity",
    "nt_xent_loss",
    "ressl_loss",
    "simsiam_loss",
    "sinkhorn_knopp",
    "swav_loss",
    "swav2_loss",
    "vicreg_loss",
    "vicreg2_loss",
]
