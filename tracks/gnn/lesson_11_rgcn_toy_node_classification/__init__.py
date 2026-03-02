"""Lesson 11 (GNN): R-GCN on a toy relational graph (node classification)."""

from .data import DataConfig, ToyRelGraph, load_toy_rel_graph
from .model import RGCN, ModelConfig

__all__ = ["DataConfig", "ToyRelGraph", "load_toy_rel_graph", "RGCN", "ModelConfig"]

