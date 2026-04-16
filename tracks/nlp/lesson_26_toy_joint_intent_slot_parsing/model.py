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
    num_intents: int = 4
    num_slot_labels: int = 7
    embed_dim: int = 64
    dropout: float = 0.1


class JointIntentSlotModel(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(
            num_embeddings=int(config.vocab_size),
            embedding_dim=int(config.embed_dim),
            padding_idx=int(config.pad_id),
        )
        self.dropout = nn.Dropout(float(config.dropout))
        self.intent_head = nn.Linear(int(config.embed_dim), int(config.num_intents))
        self.slot_head = nn.Linear(int(config.embed_dim), int(config.num_slot_labels))

    def forward(self, inputs: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")
        if attention_mask is None:
            attention_mask = (input_ids != int(self.config.pad_id)).to(torch.float32)

        token_embeddings = self.dropout(self.embedding(input_ids))
        mask = attention_mask.unsqueeze(-1).to(token_embeddings.dtype)
        pooled = (token_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        intent_logits = self.intent_head(self.dropout(pooled))
        slot_logits = self.slot_head(token_embeddings)
        return {"intent_logits": intent_logits, "slot_logits": slot_logits}


def joint_loss(
    intent_logits: torch.Tensor,
    slot_logits: torch.Tensor,
    intent_labels: torch.Tensor,
    slot_labels: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    slot_loss_weight: float = 1.0,
) -> torch.Tensor:
    intent_ce = F.cross_entropy(intent_logits, intent_labels)

    flat_slot_logits = slot_logits.reshape(-1, slot_logits.size(-1))
    flat_slot_labels = slot_labels.reshape(-1)
    flat_mask = attention_mask.reshape(-1).to(torch.float32)
    token_ce = F.cross_entropy(flat_slot_logits, flat_slot_labels, reduction="none")
    slot_ce = (token_ce * flat_mask).sum() / flat_mask.sum().clamp(min=1.0)
    return intent_ce + float(slot_loss_weight) * slot_ce


def compute_joint_metrics(
    intent_logits: torch.Tensor,
    slot_logits: torch.Tensor,
    intent_labels: torch.Tensor,
    slot_labels: torch.Tensor,
    attention_mask: torch.Tensor,
) -> dict[str, float]:
    with torch.no_grad():
        intent_preds = intent_logits.argmax(dim=1)
        intent_acc = float((intent_preds == intent_labels).to(torch.float32).mean().item())

        slot_preds = slot_logits.argmax(dim=-1)
        token_correct = (slot_preds == slot_labels).to(torch.float32) * attention_mask.to(torch.float32)
        denom = attention_mask.to(torch.float32).sum().clamp(min=1.0)
        slot_token_acc = float((token_correct.sum() / denom).item())
    return {"intent_acc": intent_acc, "slot_token_acc": slot_token_acc}


__all__ = [
    "JointIntentSlotModel",
    "ModelConfig",
    "compute_joint_metrics",
    "joint_loss",
]
