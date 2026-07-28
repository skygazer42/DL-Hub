from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


def _masked_mean_pool(x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    weights = attention_mask.to(torch.float32).unsqueeze(-1)
    return (x * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    pad_id: int
    embed_dim: int = 24
    hidden_dim: int = 32
    num_classes: int = 4
    dropout: float = 0.1


class AdversarialTextClassifier(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.embedding = nn.Embedding(
            int(cfg.vocab_size),
            int(cfg.embed_dim),
            padding_idx=int(cfg.pad_id),
        )
        self.classifier = nn.Sequential(
            nn.Linear(int(cfg.embed_dim), int(cfg.hidden_dim)),
            nn.ReLU(),
            nn.Dropout(float(cfg.dropout)),
            nn.Linear(int(cfg.hidden_dim), int(cfg.num_classes)),
        )

    def _encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        tokens = self.embedding(input_ids)
        pooled = _masked_mean_pool(tokens, attention_mask)
        return self.classifier(pooled)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {
            "clean_logits": self._encode(batch["input_ids"], batch["attention_mask"]),
            "adversarial_logits": self._encode(
                batch["adversarial_input_ids"], batch["adversarial_attention_mask"]
            ),
        }


def robust_classification_loss(
    clean_logits: torch.Tensor,
    adversarial_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    consistency_weight: float = 0.25,
) -> tuple[torch.Tensor, dict[str, float]]:
    clean_ce = torch.nn.functional.cross_entropy(clean_logits, labels)
    adv_ce = torch.nn.functional.cross_entropy(adversarial_logits, labels)
    clean_probs = torch.softmax(clean_logits.detach(), dim=-1)
    adv_probs = torch.softmax(adversarial_logits, dim=-1)
    consistency_loss = torch.mean((adv_probs - clean_probs) ** 2)
    total_loss = clean_ce + adv_ce + float(consistency_weight) * consistency_loss
    return total_loss, {
        "clean_ce_loss": float(clean_ce.item()),
        "adv_ce_loss": float(adv_ce.item()),
        "consistency_loss": float(consistency_loss.item()),
    }


def classification_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    pred = logits.argmax(dim=-1)
    return float((pred == labels).to(torch.float32).mean().item())


__all__ = [
    "AdversarialTextClassifier",
    "ModelConfig",
    "classification_accuracy",
    "robust_classification_loss",
]
