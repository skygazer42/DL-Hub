from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    pad_id: int
    embed_dim: int = 64
    hidden_dim: int = 32
    num_topics: int = 4
    dropout: float = 0.1


class TopicModelingModel(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.embedding = nn.Embedding(
            int(cfg.vocab_size),
            int(cfg.embed_dim),
            padding_idx=int(cfg.pad_id),
        )
        self.encoder = nn.Sequential(
            nn.Linear(int(cfg.embed_dim), int(cfg.hidden_dim)),
            nn.ReLU(),
            nn.Dropout(float(cfg.dropout)),
        )
        self.topic_head = nn.Linear(int(cfg.hidden_dim), int(cfg.num_topics))
        self.topic_word_logits = nn.Parameter(torch.zeros(int(cfg.num_topics), int(cfg.vocab_size)))
        nn.init.normal_(self.topic_word_logits, mean=0.0, std=0.02)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        embeddings = self.embedding(batch["input_ids"])
        mask = batch["attention_mask"].unsqueeze(-1)
        pooled = (embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        hidden = self.encoder(pooled)
        topic_probs = torch.softmax(self.topic_head(hidden), dim=-1)
        reconstruction_logits = topic_probs @ self.topic_word_logits
        return {"topic_probs": topic_probs, "reconstruction_logits": reconstruction_logits}


def topic_modeling_loss(
    reconstruction_logits: torch.Tensor,
    bow_targets: torch.Tensor,
    topic_probs: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    recon_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        reconstruction_logits,
        bow_targets.to(torch.float32),
    )
    entropy_loss = -(topic_probs.clamp(min=1e-8).log() * topic_probs).sum(dim=-1).mean()
    total_loss = recon_loss + 0.05 * entropy_loss
    return total_loss, {"recon_loss": float(recon_loss.item()), "entropy_loss": float(entropy_loss.item())}


def topic_accuracy(topic_probs: torch.Tensor, topic_labels: torch.Tensor) -> float:
    pred = topic_probs.argmax(dim=-1)
    return float((pred == topic_labels).to(torch.float32).mean().item())


__all__ = ["ModelConfig", "TopicModelingModel", "topic_accuracy", "topic_modeling_loss"]

