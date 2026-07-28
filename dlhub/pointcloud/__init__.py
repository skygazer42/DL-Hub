"""Point cloud utilities and models (no downloads, CPU-friendly).

This module provides a local point-cloud architecture zoo similar to `dlhub.vision.local_zoo`,
focused on simple, readable implementations that can be exercised on synthetic synthetic datasets.
"""

from .gaussian_splatting_zoo import (
    BuildConfig as GaussianSplattingBuildConfig,
    UnknownLocalArch as UnknownGaussianSplattingArch,
    build_local_model as build_gaussian_splatting_model,
    list_local_arches as list_gaussian_splatting_arches,
)
from .open_vocabulary_3d_zoo import (
    BuildConfig as OpenVocabulary3DBuildConfig,
    UnknownLocalArch as UnknownOpenVocabulary3DArch,
    build_local_model as build_open_vocabulary_3d_model,
    list_local_arches as list_open_vocabulary_3d_arches,
)
from .pointcloud_completion_zoo import (
    BuildConfig as PointCloudCompletionBuildConfig,
    UnknownLocalArch as UnknownPointCloudCompletionArch,
    build_local_model as build_pointcloud_completion_model,
    list_local_arches as list_pointcloud_completion_arches,
)
from .pointcloud_anomaly_detection_zoo import (
    BuildConfig as PointCloudAnomalyDetectionBuildConfig,
    UnknownLocalArch as UnknownPointCloudAnomalyDetectionArch,
    build_local_model as build_pointcloud_anomaly_detection_model,
    list_local_arches as list_pointcloud_anomaly_detection_arches,
)
from .pointcloud_forecasting_zoo import (
    BuildConfig as PointCloudForecastingBuildConfig,
    UnknownLocalArch as UnknownPointCloudForecastingArch,
    build_local_model as build_pointcloud_forecasting_model,
    list_local_arches as list_pointcloud_forecasting_arches,
)
from .pointcloud_upsampling_zoo import (
    BuildConfig as PointCloudUpsamplingBuildConfig,
    UnknownLocalArch as UnknownPointCloudUpsamplingArch,
    build_local_model as build_pointcloud_upsampling_model,
    list_local_arches as list_pointcloud_upsampling_arches,
)
from .scene_flow_zoo import (
    BuildConfig as SceneFlowBuildConfig,
    UnknownLocalArch as UnknownSceneFlowArch,
    build_local_model as build_scene_flow_model,
    list_local_arches as list_scene_flow_arches,
)
from .shape_correspondence_3d_zoo import (
    BuildConfig as ShapeCorrespondence3DBuildConfig,
    UnknownLocalArch as UnknownShapeCorrespondence3DArch,
    build_local_model as build_shape_correspondence_3d_model,
    list_local_arches as list_shape_correspondence_3d_arches,
)
from .local_zoo import BuildConfig, UnknownLocalArch, build_local_model, list_local_arches

__all__ = [
    "BuildConfig",
    "GaussianSplattingBuildConfig",
    "OpenVocabulary3DBuildConfig",
    "PointCloudAnomalyDetectionBuildConfig",
    "PointCloudCompletionBuildConfig",
    "PointCloudForecastingBuildConfig",
    "PointCloudUpsamplingBuildConfig",
    "SceneFlowBuildConfig",
    "ShapeCorrespondence3DBuildConfig",
    "UnknownLocalArch",
    "UnknownGaussianSplattingArch",
    "UnknownOpenVocabulary3DArch",
    "UnknownPointCloudAnomalyDetectionArch",
    "UnknownPointCloudCompletionArch",
    "UnknownPointCloudForecastingArch",
    "UnknownPointCloudUpsamplingArch",
    "UnknownSceneFlowArch",
    "UnknownShapeCorrespondence3DArch",
    "build_gaussian_splatting_model",
    "build_open_vocabulary_3d_model",
    "build_pointcloud_anomaly_detection_model",
    "build_pointcloud_completion_model",
    "build_pointcloud_forecasting_model",
    "build_pointcloud_upsampling_model",
    "build_scene_flow_model",
    "build_shape_correspondence_3d_model",
    "build_local_model",
    "list_gaussian_splatting_arches",
    "list_open_vocabulary_3d_arches",
    "list_pointcloud_anomaly_detection_arches",
    "list_pointcloud_completion_arches",
    "list_pointcloud_forecasting_arches",
    "list_pointcloud_upsampling_arches",
    "list_scene_flow_arches",
    "list_shape_correspondence_3d_arches",
    "list_local_arches",
]
