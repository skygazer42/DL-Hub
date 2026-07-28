from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    pad_id: int
    num_cuisine_states: int = 4
    num_area_states: int = 4
    num_party_states: int = 4
    embed_dim: int = 64
    dropout: float = 0.1


class DialogStateTracker(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(
            num_embeddings=int(config.vocab_size),
            embedding_dim=int(config.embed_dim),
            padding_idx=int(config.pad_id),
        )
        self.dropout = nn.Dropout(float(config.dropout))
        hidden_dim = int(config.embed_dim)
        self.cuisine_head = nn.Linear(hidden_dim, int(config.num_cuisine_states))
        self.area_head = nn.Linear(hidden_dim, int(config.num_area_states))
        self.party_head = nn.Linear(hidden_dim, int(config.num_party_states))

    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")
        if attention_mask is None:
            attention_mask = (input_ids != int(self.config.pad_id)).to(torch.float32)

        token_embeddings = self.dropout(self.embedding(input_ids))
        mask = attention_mask.unsqueeze(-1).to(token_embeddings.dtype)
        pooled = (token_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        pooled = self.dropout(pooled)
        return {
            "cuisine_logits": self.cuisine_head(pooled),
            "area_logits": self.area_head(pooled),
            "party_logits": self.party_head(pooled),
        }


def dialog_state_loss(
    cuisine_logits: torch.Tensor,
    area_logits: torch.Tensor,
    party_logits: torch.Tensor,
    cuisine_labels: torch.Tensor,
    area_labels: torch.Tensor,
    party_labels: torch.Tensor,
) -> torch.Tensor:
    return (
        F.cross_entropy(cuisine_logits, cuisine_labels)
        + F.cross_entropy(area_logits, area_labels)
        + F.cross_entropy(party_logits, party_labels)
    ) / 3.0


def compute_state_metrics(
    cuisine_logits: torch.Tensor,
    area_logits: torch.Tensor,
    party_logits: torch.Tensor,
    cuisine_labels: torch.Tensor,
    area_labels: torch.Tensor,
    party_labels: torch.Tensor,
) -> dict[str, float]:
    with torch.no_grad():
        cuisine_pred = cuisine_logits.argmax(dim=1)
        area_pred = area_logits.argmax(dim=1)
        party_pred = party_logits.argmax(dim=1)

        cuisine_acc = (cuisine_pred == cuisine_labels).to(torch.float32)
        area_acc = (area_pred == area_labels).to(torch.float32)
        party_acc = (party_pred == party_labels).to(torch.float32)

        slot_acc = torch.stack([cuisine_acc, area_acc, party_acc], dim=0).mean()
        joint_goal_acc = (cuisine_acc * area_acc * party_acc).mean()
    return {"slot_acc": float(slot_acc.item()), "joint_goal_acc": float(joint_goal_acc.item())}


__all__ = [
    "DialogStateTracker",
    "ModelConfig",
    "compute_state_metrics",
    "dialog_state_loss",
]
