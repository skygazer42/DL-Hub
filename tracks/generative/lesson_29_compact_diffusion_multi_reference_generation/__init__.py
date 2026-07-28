"""Lesson 29: compact diffusion for multi-reference generation."""

from .data import DataConfig, SyntheticMultiReferenceGenerationDataset, get_dataloaders
from .model import DiffusionSchedule, ModelConfig, CompactMultiReferenceDiffusionModel, q_sample
from .train import TrainConfig, run_training

__all__ = [
    "DataConfig",
    "DiffusionSchedule",
    "ModelConfig",
    "CompactMultiReferenceDiffusionModel",
    "SyntheticMultiReferenceGenerationDataset",
    "TrainConfig",
    "get_dataloaders",
    "q_sample",
    "run_training",
]
