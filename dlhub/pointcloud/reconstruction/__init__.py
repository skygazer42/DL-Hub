"""Point cloud reconstruction models (pure torch, toy-first).

Conventions:
- One algorithm family per file (variants live in that file).
- Each file exposes a `build_<name>_...(...)` factory and a `__main__` smoke test.
"""

from .atlasnet import AtlasNetAutoEncoder, build_atlasnet_autoencoder
from .foldingnet import FoldingNetAutoEncoder, build_foldingnet_autoencoder
from .grnet import GRNetAutoEncoder, build_grnet_autoencoder
from .morphing_recon import MorphingReconAutoEncoder, build_morphing_recon_autoencoder
from .pcn import PCNAutoEncoder, build_pcn_autoencoder
from .pointcaps_recon import PointcapsReconAutoEncoder, build_pointcaps_recon_autoencoder
from .pointnet_ae import PointNetAutoEncoder, build_pointnet_autoencoder
from .pointr import PointrAutoEncoder, build_pointr_autoencoder
from .snowflake_recon import SnowflakeReconAutoEncoder, build_snowflake_recon_autoencoder
from .spare_net import SpareNetAutoEncoder, build_spare_net_autoencoder
from .topnet import TopNetAutoEncoder, build_topnet_autoencoder

__all__ = [
    "AtlasNetAutoEncoder",
    "FoldingNetAutoEncoder",
    "GRNetAutoEncoder",
    "MorphingReconAutoEncoder",
    "PCNAutoEncoder",
    "PointcapsReconAutoEncoder",
    "PointNetAutoEncoder",
    "PointrAutoEncoder",
    "SnowflakeReconAutoEncoder",
    "SpareNetAutoEncoder",
    "TopNetAutoEncoder",
    "build_atlasnet_autoencoder",
    "build_foldingnet_autoencoder",
    "build_grnet_autoencoder",
    "build_morphing_recon_autoencoder",
    "build_pcn_autoencoder",
    "build_pointcaps_recon_autoencoder",
    "build_pointnet_autoencoder",
    "build_pointr_autoencoder",
    "build_snowflake_recon_autoencoder",
    "build_spare_net_autoencoder",
    "build_topnet_autoencoder",
]
