"""Lesson 35: toy shape correspondence in 3D point clouds."""

from .data import DataConfig, ToyShapeCorrespondenceDataset, get_dataloaders
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
    "ToyShapeCorrespondenceDataset",
    "TrainConfig",
    "build_model",
    "correspondence_accuracy",
    "correspondence_loss",
    "get_dataloaders",
    "list_supported_arches",
    "run_training",
]
