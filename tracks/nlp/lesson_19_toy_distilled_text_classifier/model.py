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
    student_embed_dim: int = 24
    teacher_embed_dim: int = 48
    num_classes: int = 4
    dropout: float = 0.1


class DistilledTextClassifier(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.student_embedding = nn.Embedding(
            int(cfg.vocab_size),
            int(cfg.student_embed_dim),
            padding_idx=int(cfg.pad_id),
        )
        self.teacher_embedding = nn.Embedding(
            int(cfg.vocab_size),
            int(cfg.teacher_embed_dim),
            padding_idx=int(cfg.pad_id),
        )
        self.student_head = nn.Sequential(
            nn.Linear(int(cfg.student_embed_dim), int(cfg.student_embed_dim)),
            nn.ReLU(),
            nn.Dropout(float(cfg.dropout)),
            nn.Linear(int(cfg.student_embed_dim), int(cfg.num_classes)),
        )
        self.teacher_head = nn.Sequential(
            nn.Linear(int(cfg.teacher_embed_dim), int(cfg.teacher_embed_dim)),
            nn.ReLU(),
            nn.Dropout(float(cfg.dropout)),
            nn.Linear(int(cfg.teacher_embed_dim), int(cfg.num_classes)),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        student_tokens = self.student_embedding(batch["input_ids"])
        teacher_tokens = self.teacher_embedding(batch["input_ids"])
        student_pooled = _masked_mean_pool(student_tokens, batch["attention_mask"])
        teacher_pooled = _masked_mean_pool(teacher_tokens, batch["attention_mask"])
        return {
            "student_logits": self.student_head(student_pooled),
            "teacher_logits": self.teacher_head(teacher_pooled),
        }


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    temperature: float = 2.0,
    alpha: float = 0.7,
) -> tuple[torch.Tensor, dict[str, float]]:
    teacher_ce = torch.nn.functional.cross_entropy(teacher_logits, labels)
    student_ce = torch.nn.functional.cross_entropy(student_logits, labels)
    ce_loss = student_ce + 0.5 * teacher_ce

    student_log_probs = torch.nn.functional.log_softmax(student_logits / float(temperature), dim=-1)
    teacher_probs = torch.nn.functional.softmax(teacher_logits.detach() / float(temperature), dim=-1)
    distill_loss = torch.nn.functional.kl_div(
        student_log_probs,
        teacher_probs,
        reduction="batchmean",
    ) * (float(temperature) ** 2)
    total_loss = ce_loss + float(alpha) * distill_loss
    return total_loss, {"distill_loss": float(distill_loss.item()), "ce_loss": float(ce_loss.item())}


def classification_accuracy(student_logits: torch.Tensor, labels: torch.Tensor) -> float:
    pred = student_logits.argmax(dim=-1)
    return float((pred == labels).to(torch.float32).mean().item())


__all__ = [
    "DistilledTextClassifier",
    "ModelConfig",
    "classification_accuracy",
    "distillation_loss",
]

