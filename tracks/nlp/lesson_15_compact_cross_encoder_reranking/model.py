from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from tracks.nlp.lesson_02_compact_text_classification_transformer.model import TransformerEncoderBlock


def _masked_mean_pool(x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    weights = attention_mask.to(torch.float32).unsqueeze(-1)
    return (x * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    pad_id: int
    max_length: int
    embed_dim: int = 64
    num_heads: int = 4
    num_layers: int = 2
    ff_dim: int = 128
    dropout: float = 0.1


class CrossEncoderReranker(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.token_embed = nn.Embedding(
            num_embeddings=int(cfg.vocab_size),
            embedding_dim=int(cfg.embed_dim),
            padding_idx=int(cfg.pad_id),
        )
        self.pos_embed = nn.Embedding(int(cfg.max_length), int(cfg.embed_dim))
        self.dropout = nn.Dropout(float(cfg.dropout))
        self.blocks = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    embed_dim=int(cfg.embed_dim),
                    num_heads=int(cfg.num_heads),
                    ff_dim=int(cfg.ff_dim),
                    dropout=float(cfg.dropout),
                )
                for _ in range(int(cfg.num_layers))
            ]
        )
        self.ln = nn.LayerNorm(int(cfg.embed_dim))
        self.score_head = nn.Linear(int(cfg.embed_dim), 1)

    def _score_pair(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        if seq_len != int(self.cfg.max_length):
            raise ValueError(
                f"Expected max_length={int(self.cfg.max_length)} tokens, got sequence length {int(seq_len)}"
            )
        pos = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, seq_len)
        x = self.token_embed(input_ids) + self.pos_embed(pos)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x, attention_mask=attention_mask)
        x = self.ln(x)
        pooled = _masked_mean_pool(x, attention_mask)
        return self.score_head(pooled).squeeze(-1)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        positive_scores = self._score_pair(
            batch["positive_input_ids"],
            batch["positive_attention_mask"],
        )
        negative_scores = self._score_pair(
            batch["negative_input_ids"],
            batch["negative_attention_mask"],
        )
        return {
            "positive_scores": positive_scores,
            "negative_scores": negative_scores,
        }


def reranking_accuracy(positive_scores: torch.Tensor, negative_scores: torch.Tensor) -> float:
    return float((positive_scores > negative_scores).to(torch.float32).mean().item())


__all__ = ["CrossEncoderReranker", "ModelConfig", "reranking_accuracy"]
