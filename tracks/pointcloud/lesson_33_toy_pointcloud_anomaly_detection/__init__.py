"""Lesson 33: toy pointcloud anomaly detection."""

from .data import DataConfig, SyntheticPointCloudAnomalyDataset, get_dataloaders
from .model import (
    ModelConfig,
    anomaly_accuracy,
    anomaly_loss,
    build_model,
    list_supported_arches,
)
from .train import TrainConfig, run_training

__all__ = [
    "DataConfig",
    "ModelConfig",
    "SyntheticPointCloudAnomalyDataset",
    "TrainConfig",
    "anomaly_accuracy",
    "anomaly_loss",
    "build_model",
    "get_dataloaders",
    "list_supported_arches",
    "run_training",
]
