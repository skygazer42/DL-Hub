from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    num_nodes: int
    embed_dim: int = 16
    order: int = 2  # 1 or 2


class LINE(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        if cfg.order not in (1, 2):
            raise ValueError("order must be 1 or 2")

        self.cfg = cfg
        self.node_embeddings = nn.Embedding(int(cfg.num_nodes), int(cfg.embed_dim))
        nn.init.uniform_(self.node_embeddings.weight, a=-0.5 / cfg.embed_dim, b=0.5 / cfg.embed_dim)

        self.context_embeddings: nn.Embedding | None = None
        if int(cfg.order) == 2:
            self.context_embeddings = nn.Embedding(int(cfg.num_nodes), int(cfg.embed_dim))
            nn.init.uniform_(
                self.context_embeddings.weight, a=-0.5 / cfg.embed_dim, b=0.5 / cfg.embed_dim
            )

    def _target_table(self) -> nn.Embedding:
        return self.context_embeddings if self.context_embeddings is not None else self.node_embeddings

    def loss(self, *, src: torch.Tensor, dst: torch.Tensor, neg_dst: torch.Tensor) -> torch.Tensor:
        """Compute LINE loss for a batch.

        src: (B,)
        dst: (B,)
        neg_dst: (B, K)
        """

        src = src.to(torch.long)
        dst = dst.to(torch.long)
        neg_dst = neg_dst.to(torch.long)

        u = self.node_embeddings(src)  # (B, D)
        v = self._target_table()(dst)  # (B, D)
        neg_v = self._target_table()(neg_dst)  # (B, K, D)

        pos_score = (u * v).sum(dim=1)  # (B,)
        neg_score = (u.unsqueeze(1) * neg_v).sum(dim=-1)  # (B, K)

        pos = torch.nn.functional.logsigmoid(pos_score)
        neg = torch.nn.functional.logsigmoid(-neg_score).sum(dim=1)
        return -(pos + neg).mean()


__all__ = ["LINE", "ModelConfig"]

