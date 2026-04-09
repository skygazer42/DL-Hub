"""Point cloud utilities and models (no downloads, CPU-friendly).

This module provides a local point-cloud architecture zoo similar to `dlhub.vision.local_zoo`,
focused on simple, readable implementations that can be exercised on synthetic toy datasets.
"""

from .gaussian_splatting_zoo import (
    BuildConfig as GaussianSplattingBuildConfig,
    UnknownLocalArch as UnknownGaussianSplattingArch,
    build_local_model as build_gaussian_splatting_model,
    list_local_arches as list_gaussian_splatting_arches,
)
from .local_zoo import BuildConfig, UnknownLocalArch, build_local_model, list_local_arches

__all__ = [
    "BuildConfig",
    "GaussianSplattingBuildConfig",
    "UnknownLocalArch",
    "UnknownGaussianSplattingArch",
    "build_gaussian_splatting_model",
    "build_local_model",
    "list_gaussian_splatting_arches",
    "list_local_arches",
]
