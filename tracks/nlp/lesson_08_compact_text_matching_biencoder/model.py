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
    proj_dim: int = 64
    dropout: float = 0.1


class MeanPoolEncoder(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.pad_id = int(config.pad_id)
        self.embedding = nn.Embedding(
            num_embeddings=int(config.vocab_size),
            embedding_dim=int(config.embed_dim),
            padding_idx=int(config.pad_id),
        )
        self.proj = nn.Linear(int(config.embed_dim), int(config.proj_dim))
        self.dropout = nn.Dropout(float(config.dropout))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
        if attention_mask is None:
            attention_mask = (input_ids != self.pad_id).to(torch.float32)
        emb = self.embedding(input_ids)
        mask = attention_mask.unsqueeze(-1).to(emb.dtype)
        pooled = (emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        projected = self.proj(self.dropout(pooled))
        return F.normalize(projected, dim=-1)


class BiEncoderTextMatcher(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = MeanPoolEncoder(config)

    def forward(self, inputs: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        query_embeddings = self.encoder(
            inputs["query_input_ids"],
            inputs.get("query_attention_mask"),
        )
        doc_embeddings = self.encoder(
            inputs["doc_input_ids"],
            inputs.get("doc_attention_mask"),
        )
        sim_matrix = query_embeddings @ doc_embeddings.transpose(0, 1)
        pair_logits = torch.diagonal(sim_matrix)
        return {
            "query_embeddings": query_embeddings,
            "doc_embeddings": doc_embeddings,
            "pair_logits": pair_logits,
            "sim_matrix": sim_matrix,
        }


def contrastive_retrieval_loss(sim_matrix: torch.Tensor) -> torch.Tensor:
    targets = torch.arange(sim_matrix.shape[0], device=sim_matrix.device)
    q2d = F.cross_entropy(sim_matrix, targets)
    d2q = F.cross_entropy(sim_matrix.transpose(0, 1), targets)
    return 0.5 * (q2d + d2q)


def retrieval_accuracy(
    *,
    pair_logits: torch.Tensor,
    sim_matrix: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[float, float]:
    pred_pairs = (torch.sigmoid(pair_logits) >= 0.5).to(labels.dtype)
    pair_acc = float((pred_pairs == labels).to(torch.float32).mean().item())
    targets = torch.arange(sim_matrix.shape[0], device=sim_matrix.device)
    retrieved = sim_matrix.argmax(dim=1)
    retrieval_acc = float((retrieved == targets).to(torch.float32).mean().item())
    return pair_acc, retrieval_acc


__all__ = [
    "BiEncoderTextMatcher",
    "ModelConfig",
    "contrastive_retrieval_loss",
    "retrieval_accuracy",
]
