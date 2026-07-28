from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    pad_id: int
    embed_dim: int = 64
    dropout: float = 0.1


class DialogResponseSelector(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(
            num_embeddings=int(config.vocab_size),
            embedding_dim=int(config.embed_dim),
            padding_idx=int(config.pad_id),
        )
        self.dropout = nn.Dropout(float(config.dropout))

    def _pool(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        embeddings = self.dropout(self.embedding(input_ids))
        mask = attention_mask.unsqueeze(-1).to(embeddings.dtype)
        return (embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

    def forward(self, inputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        context_ids = inputs["context_ids"]
        context_mask = inputs.get("context_attention_mask")
        if context_mask is None:
            context_mask = (context_ids != int(self.config.pad_id)).to(torch.float32)

        candidate_ids = inputs["candidate_ids"]
        candidate_mask = inputs.get("candidate_attention_mask")
        if candidate_mask is None:
            candidate_mask = (candidate_ids != int(self.config.pad_id)).to(torch.float32)

        batch_size, num_candidates, seq_len = candidate_ids.shape
        context_vec = self._pool(context_ids, context_mask)
        flat_candidates = candidate_ids.view(batch_size * num_candidates, seq_len)
        flat_candidate_mask = candidate_mask.view(batch_size * num_candidates, seq_len)
        candidate_vec = self._pool(flat_candidates, flat_candidate_mask)
        candidate_vec = candidate_vec.view(batch_size, num_candidates, -1)
        return torch.einsum("bd,bcd->bc", context_vec, candidate_vec)


def response_selection_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, labels)


def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    with torch.no_grad():
        preds = logits.argmax(dim=1)
        return float((preds == labels).to(torch.float32).mean().item())


__all__ = ["DialogResponseSelector", "ModelConfig", "compute_accuracy", "response_selection_loss"]
