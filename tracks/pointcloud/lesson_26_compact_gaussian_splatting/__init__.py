"""Lesson 26: compact Gaussian splatting for point clouds."""

from .data import DataConfig, SyntheticGaussianSplattingDataset, get_dataloaders
from .model import ModelConfig, CompactGaussianSplattingModel, gaussian_splatting_loss, render_gaussians

__all__ = [
    "DataConfig",
    "ModelConfig",
    "SyntheticGaussianSplattingDataset",
    "CompactGaussianSplattingModel",
    "gaussian_splatting_loss",
    "get_dataloaders",
    "render_gaussians",
]
