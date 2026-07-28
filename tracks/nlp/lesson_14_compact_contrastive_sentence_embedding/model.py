from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    pad_id: int
    embed_dim: int = 64
    proj_dim: int = 32
    dropout: float = 0.1


def nt_xent_loss(sim_matrix: torch.Tensor, *, temperature: float = 0.1) -> torch.Tensor:
    if sim_matrix.ndim != 2 or sim_matrix.shape[0] != sim_matrix.shape[1]:
        raise ValueError(f"Expected square sim matrix (B, B), got {tuple(sim_matrix.shape)}")
    if sim_matrix.shape[0] < 2:
        raise ValueError("NT-Xent requires batch size >= 2 for contrastive negatives.")

    logits = sim_matrix / float(temperature)
    labels = torch.arange(sim_matrix.shape[0], device=sim_matrix.device)
    loss12 = F.cross_entropy(logits, labels)
    loss21 = F.cross_entropy(logits.transpose(0, 1), labels)
    return 0.5 * (loss12 + loss21)


def contrastive_accuracy(sim_matrix: torch.Tensor) -> float:
    if sim_matrix.ndim != 2 or sim_matrix.shape[0] != sim_matrix.shape[1]:
        raise ValueError(f"Expected square sim matrix (B, B), got {tuple(sim_matrix.shape)}")
    if sim_matrix.shape[0] == 0:
        return 0.0

    with torch.no_grad():
        labels = torch.arange(sim_matrix.shape[0], device=sim_matrix.device)
        acc12 = sim_matrix.argmax(dim=1).eq(labels).to(torch.float32).mean()
        acc21 = sim_matrix.argmax(dim=0).eq(labels).to(torch.float32).mean()
        return float(0.5 * (acc12 + acc21))


class ContrastiveSentenceEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.token_embed = nn.Embedding(
            num_embeddings=int(cfg.vocab_size),
            embedding_dim=int(cfg.embed_dim),
            padding_idx=int(cfg.pad_id),
        )
        self.proj = nn.Sequential(
            nn.Linear(int(cfg.embed_dim), int(cfg.embed_dim)),
            nn.ReLU(),
            nn.Dropout(p=float(cfg.dropout)),
            nn.Linear(int(cfg.embed_dim), int(cfg.proj_dim)),
        )

    def _encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        token_embeddings = self.token_embed(input_ids)
        mask = attention_mask.unsqueeze(-1)
        pooled = (token_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        projected = self.proj(pooled)
        return F.normalize(projected, p=2, dim=-1)

    def forward(self, batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        view1_embeddings = self._encode(batch["view1_input_ids"], batch["view1_attention_mask"])
        view2_embeddings = self._encode(batch["view2_input_ids"], batch["view2_attention_mask"])
        sim_matrix = view1_embeddings @ view2_embeddings.transpose(0, 1)
        return {
            "view1_embeddings": view1_embeddings,
            "view2_embeddings": view2_embeddings,
            "sim_matrix": sim_matrix,
        }


__all__ = ["ContrastiveSentenceEncoder", "ModelConfig", "contrastive_accuracy", "nt_xent_loss"]
