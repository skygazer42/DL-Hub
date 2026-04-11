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
from .pointcloud_completion_zoo import (
    BuildConfig as PointCloudCompletionBuildConfig,
    UnknownLocalArch as UnknownPointCloudCompletionArch,
    build_local_model as build_pointcloud_completion_model,
    list_local_arches as list_pointcloud_completion_arches,
)
from .scene_flow_zoo import (
    BuildConfig as SceneFlowBuildConfig,
    UnknownLocalArch as UnknownSceneFlowArch,
    build_local_model as build_scene_flow_model,
    list_local_arches as list_scene_flow_arches,
)
from .local_zoo import BuildConfig, UnknownLocalArch, build_local_model, list_local_arches

__all__ = [
    "BuildConfig",
    "GaussianSplattingBuildConfig",
    "PointCloudCompletionBuildConfig",
    "SceneFlowBuildConfig",
    "UnknownLocalArch",
    "UnknownGaussianSplattingArch",
    "UnknownPointCloudCompletionArch",
    "UnknownSceneFlowArch",
    "build_gaussian_splatting_model",
    "build_pointcloud_completion_model",
    "build_scene_flow_model",
    "build_local_model",
    "list_gaussian_splatting_arches",
    "list_pointcloud_completion_arches",
    "list_scene_flow_arches",
    "list_local_arches",
]
