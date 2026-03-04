"""Self-supervised learning for point clouds (pure torch, toy-first)."""

from .barlowtwins import BarlowTwinsPointNet, barlow_twins_loss, build_barlowtwins_pointnet
from .byol import BYOLPointNet, build_byol_pointnet, byol_loss, cosine_similarity_loss
from .dino import DINOPointNet, build_dino_pointnet, dino_loss
from .dinov2 import DINOV2PointMAE, build_dinov2_pointmae, dino_cross_view_loss, ibot_patch_loss
from .ijepa import IJEPAPointMAE, build_ijepa_pointmae, ijepa_patch_loss
from .msn import MSNPointMAE, build_msn_pointmae, msn_loss
from .moco import MoCoPointNet, build_moco_pointnet, moco_logits
from .simclr import SimCLRPointNet, build_simclr_pointnet, nt_xent_loss
from .simsiam import SimSiamPointNet, build_simsiam_pointnet, negative_cosine_similarity, simsiam_loss
from .swav import SwAVPointNet, build_swav_pointnet, sinkhorn_knopp, swav_loss
from .pointmae import PointMAEPretrainer, build_pointmae_pretrainer
from .vicreg import VICRegPointNet, build_vicreg_pointnet, vicreg_loss

__all__ = [
    "BarlowTwinsPointNet",
    "BYOLPointNet",
    "DINOPointNet",
    "DINOV2PointMAE",
    "IJEPAPointMAE",
    "MSNPointMAE",
    "MoCoPointNet",
    "PointMAEPretrainer",
    "SimCLRPointNet",
    "SimSiamPointNet",
    "SwAVPointNet",
    "VICRegPointNet",
    "barlow_twins_loss",
    "build_dinov2_pointmae",
    "build_barlowtwins_pointnet",
    "build_byol_pointnet",
    "build_dino_pointnet",
    "build_ijepa_pointmae",
    "build_msn_pointmae",
    "build_moco_pointnet",
    "build_pointmae_pretrainer",
    "build_simclr_pointnet",
    "build_simsiam_pointnet",
    "build_swav_pointnet",
    "build_vicreg_pointnet",
    "byol_loss",
    "cosine_similarity_loss",
    "dino_cross_view_loss",
    "dino_loss",
    "ibot_patch_loss",
    "ijepa_patch_loss",
    "msn_loss",
    "moco_logits",
    "negative_cosine_similarity",
    "nt_xent_loss",
    "simsiam_loss",
    "sinkhorn_knopp",
    "swav_loss",
    "vicreg_loss",
]
