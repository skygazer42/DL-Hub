from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    num_items: int
    embed_dim: int = 64
    num_neighbors: int = 8
    normalize: bool = True


class PinSAGEItemEncoder(nn.Module):
    """A minimal PinSAGE-style item encoder (one-hop GraphSAGE aggregation)."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.item_embeddings = nn.Embedding(int(cfg.num_items), int(cfg.embed_dim))
        nn.init.uniform_(self.item_embeddings.weight, a=-0.5 / cfg.embed_dim, b=0.5 / cfg.embed_dim)

        self.proj_self = nn.Linear(int(cfg.embed_dim), int(cfg.embed_dim), bias=True)
        self.proj_neigh = nn.Linear(int(cfg.embed_dim), int(cfg.embed_dim), bias=False)
        self.act = nn.ReLU()

    def encode(self, *, item_ids: torch.Tensor, neighbors: torch.Tensor) -> torch.Tensor:
        """Encode item ids with their sampled neighbor item ids.

        item_ids: (B,)
        neighbors: (B, K) padded with -1
        """

        item_ids = item_ids.to(torch.long)
        neighbors = neighbors.to(torch.long)

        e_self = self.item_embeddings(item_ids)  # (B, D)

        # Mask padding neighbors (-1).
        mask = (neighbors >= 0).to(e_self.dtype)  # (B, K)
        safe_neighbors = neighbors.clamp(min=0)
        e_neigh = self.item_embeddings(safe_neighbors)  # (B, K, D)
        e_neigh = e_neigh * mask.unsqueeze(-1)
        denom = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        neigh_mean = e_neigh.sum(dim=1) / denom  # (B, D)

        h = self.proj_self(e_self) + self.proj_neigh(neigh_mean)
        h = self.act(h)
        if self.cfg.normalize:
            h = torch.nn.functional.normalize(h, p=2, dim=1, eps=1e-12)
        return h

    def loss(self, *, center: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor) -> torch.Tensor:
        """Negative-sampling loss for item-item similarity.

        center: (B, D)
        pos: (B, D)
        neg: (B, K, D)
        """

        pos_score = (center * pos).sum(dim=1)  # (B,)
        neg_score = (center.unsqueeze(1) * neg).sum(dim=-1)  # (B, K)
        pos_term = torch.nn.functional.logsigmoid(pos_score)
        neg_term = torch.nn.functional.logsigmoid(-neg_score).sum(dim=1)
        return -(pos_term + neg_term).mean()


__all__ = ["PinSAGEItemEncoder", "ModelConfig"]

