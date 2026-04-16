"""Lesson 38: synthetic face detection."""

from .data import DataConfig, SyntheticFaceDetectionDataset, get_dataloaders
from .model import (
    FaceDetectionConfig,
    FaceDetectionModel,
    box_l1_error_pixels,
    detection_loss,
)
from .train import TrainConfig, run_training

__all__ = [
    "DataConfig",
    "SyntheticFaceDetectionDataset",
    "get_dataloaders",
    "FaceDetectionConfig",
    "FaceDetectionModel",
    "detection_loss",
    "box_l1_error_pixels",
    "TrainConfig",
    "run_training",
]
