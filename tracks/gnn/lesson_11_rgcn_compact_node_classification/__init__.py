"""Lesson 11 (GNN): R-GCN on a synthetic relational graph (node classification)."""

from .data import DataConfig, SyntheticRelGraph, load_synthetic_rel_graph
from .model import RGCN, ModelConfig

__all__ = ["DataConfig", "SyntheticRelGraph", "load_synthetic_rel_graph", "RGCN", "ModelConfig"]
