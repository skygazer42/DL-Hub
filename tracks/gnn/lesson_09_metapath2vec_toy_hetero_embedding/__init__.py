"""Lesson 09 (GNN): metapath2vec-style embeddings on a toy heterogeneous graph."""

from .data import DataConfig, build_toy_hetero_graph, build_training_pairs
from .model import MetaPath2Vec, ModelConfig

__all__ = [
    "DataConfig",
    "build_toy_hetero_graph",
    "build_training_pairs",
    "MetaPath2Vec",
    "ModelConfig",
]
