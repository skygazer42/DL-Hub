
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    num_nodes: int
    embed_dim: int = 128
    sparse: bool = True


class MetaPath2Vec(nn.Module):
    """Skip-gram embeddings with negative sampling.

    This is a minimal metapath2vec-style model: the "metapath" part lives in the walk generator.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.u_embeddings = nn.Embedding(int(cfg.num_nodes), int(cfg.embed_dim), sparse=bool(cfg.sparse))
        self.v_embeddings = nn.Embedding(int(cfg.num_nodes), int(cfg.embed_dim), sparse=bool(cfg.sparse))

        init_range = 0.5 / float(cfg.embed_dim)
        nn.init.uniform_(self.u_embeddings.weight, a=-init_range, b=init_range)
        nn.init.constant_(self.v_embeddings.weight, 0.0)

    def loss(self, *, center: torch.Tensor, context: torch.Tensor, neg_context: torch.Tensor) -> torch.Tensor:
        center = center.to(torch.long)
        context = context.to(torch.long)
        neg_context = neg_context.to(torch.long)

        u = self.u_embeddings(center)  # (B, D)
        v = self.v_embeddings(context)  # (B, D)
        neg_v = self.v_embeddings(neg_context)  # (B, K, D)

        pos_score = (u * v).sum(dim=1)  # (B,)
        neg_score = (u.unsqueeze(1) * neg_v).sum(dim=-1)  # (B, K)

        pos = torch.nn.functional.logsigmoid(pos_score)
        neg = torch.nn.functional.logsigmoid(-neg_score).sum(dim=1)
        return -(pos + neg).mean()


__all__ = ["MetaPath2Vec", "ModelConfig"]

