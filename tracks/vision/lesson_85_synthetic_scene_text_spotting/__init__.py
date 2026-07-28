"""Lesson 85: synthetic scene text spotting (compact-first)."""

from .data import DataConfig, SyntheticSceneTextSpottingDataset, SpottingVocab, get_dataloaders
from .model import (
    ModelConfig,
    SceneTextSpotter,
    scene_text_spotting_loss,
    sequence_word_accuracy,
)
from .train import TrainConfig, run_training

__all__ = [
    "DataConfig",
    "ModelConfig",
    "SceneTextSpotter",
    "SyntheticSceneTextSpottingDataset",
    "SpottingVocab",
    "TrainConfig",
    "get_dataloaders",
    "run_training",
    "scene_text_spotting_loss",
    "sequence_word_accuracy",
]
