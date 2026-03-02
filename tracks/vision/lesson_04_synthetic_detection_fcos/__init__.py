"""Lesson 04 (Vision): synthetic anchor-free detection (FCOS-style)."""

from .data import DataConfig, get_dataloaders
from .model import TinyFCOS, ModelConfig

__all__ = ["DataConfig", "get_dataloaders", "TinyFCOS", "ModelConfig"]

