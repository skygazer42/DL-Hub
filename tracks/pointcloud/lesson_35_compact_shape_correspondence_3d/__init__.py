"""Lesson 35: compact shape correspondence in 3D point clouds."""

from .data import DataConfig, SyntheticShapeCorrespondenceDataset, get_dataloaders
from .model import (
    ModelConfig,
    build_model,
    correspondence_accuracy,
    correspondence_loss,
    list_supported_arches,
)
from .train import TrainConfig, run_training

__all__ = [
    "DataConfig",
    "ModelConfig",
    "SyntheticShapeCorrespondenceDataset",
    "TrainConfig",
    "build_model",
    "correspondence_accuracy",
    "correspondence_loss",
    "get_dataloaders",
    "list_supported_arches",
    "run_training",
]
