"""Point cloud reconstruction models (pure torch, toy-first).

Conventions:
- One algorithm family per file (variants live in that file).
- Each file exposes a `build_<name>_...(...)` factory and a `__main__` smoke test.
"""

from .atlasnet import AtlasNetAutoEncoder, build_atlasnet_autoencoder
from .clip_recon import ClipReconAutoEncoder, build_clip_recon_autoencoder
from .curve_recon import CurveReconAutoEncoder, build_curve_recon_autoencoder
from .foldingnet import FoldingNetAutoEncoder, build_foldingnet_autoencoder
from .grnet import GRNetAutoEncoder, build_grnet_autoencoder
from .hybridformer_recon import HybridformerReconAutoEncoder, build_hybridformer_recon_autoencoder
from .latent_recon import LatentReconAutoEncoder, build_latent_recon_autoencoder
from .mesh_recon import MeshReconAutoEncoder, build_mesh_recon_autoencoder
from .morphing_recon import MorphingReconAutoEncoder, build_morphing_recon_autoencoder
from .pcn import PCNAutoEncoder, build_pcn_autoencoder
from .pointcaps_recon import PointcapsReconAutoEncoder, build_pointcaps_recon_autoencoder
from .pointmamba_recon import PointmambaReconAutoEncoder, build_pointmamba_recon_autoencoder
from .pointnet_ae import PointNetAutoEncoder, build_pointnet_autoencoder
from .pointr import PointrAutoEncoder, build_pointr_autoencoder
from .prompt_recon import PromptReconAutoEncoder, build_prompt_recon_autoencoder
from .seed_recon import SeedReconAutoEncoder, build_seed_recon_autoencoder
from .seedformer_recon import SeedformerReconAutoEncoder, build_seedformer_recon_autoencoder
from .snowflake_recon import SnowflakeReconAutoEncoder, build_snowflake_recon_autoencoder
from .spare_net import SpareNetAutoEncoder, build_spare_net_autoencoder
from .token_recon import TokenReconAutoEncoder, build_token_recon_autoencoder
from .topnet import TopNetAutoEncoder, build_topnet_autoencoder

__all__ = [
    "AtlasNetAutoEncoder",
    "ClipReconAutoEncoder",
    "CurveReconAutoEncoder",
    "FoldingNetAutoEncoder",
    "GRNetAutoEncoder",
    "HybridformerReconAutoEncoder",
    "LatentReconAutoEncoder",
    "MeshReconAutoEncoder",
    "MorphingReconAutoEncoder",
    "PCNAutoEncoder",
    "PointcapsReconAutoEncoder",
    "PointmambaReconAutoEncoder",
    "PointNetAutoEncoder",
    "PointrAutoEncoder",
    "PromptReconAutoEncoder",
    "SeedReconAutoEncoder",
    "SeedformerReconAutoEncoder",
    "SnowflakeReconAutoEncoder",
    "SpareNetAutoEncoder",
    "TokenReconAutoEncoder",
    "TopNetAutoEncoder",
    "build_atlasnet_autoencoder",
    "build_clip_recon_autoencoder",
    "build_curve_recon_autoencoder",
    "build_foldingnet_autoencoder",
    "build_grnet_autoencoder",
    "build_hybridformer_recon_autoencoder",
    "build_latent_recon_autoencoder",
    "build_mesh_recon_autoencoder",
    "build_morphing_recon_autoencoder",
    "build_pcn_autoencoder",
    "build_pointcaps_recon_autoencoder",
    "build_pointmamba_recon_autoencoder",
    "build_pointnet_autoencoder",
    "build_pointr_autoencoder",
    "build_prompt_recon_autoencoder",
    "build_seed_recon_autoencoder",
    "build_seedformer_recon_autoencoder",
    "build_snowflake_recon_autoencoder",
    "build_spare_net_autoencoder",
    "build_token_recon_autoencoder",
    "build_topnet_autoencoder",
]
