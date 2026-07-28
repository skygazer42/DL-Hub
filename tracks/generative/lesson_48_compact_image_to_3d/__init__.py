"""Lesson 48: compact image-to-3D generation."""

from .data import DataConfig, SyntheticImageTo3DDataset, get_dataloaders
from .model import ModelConfig, CompactImageTo3DModel, image_to_3d_loss

__all__ = [
    "DataConfig",
    "ModelConfig",
    "SyntheticImageTo3DDataset",
    "CompactImageTo3DModel",
    "get_dataloaders",
    "image_to_3d_loss",
]
