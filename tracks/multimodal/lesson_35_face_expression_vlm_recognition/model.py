from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


class PromptEncoder(nn.Module):
    def __init__(self, *, vocab_size: int, pad_id: int, text_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(int(vocab_size), int(text_dim), padding_idx=int(pad_id))
        self.proj = nn.Linear(int(text_dim), int(hidden_dim))

    def forward(self, prompt_ids: torch.Tensor, prompt_mask: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(prompt_ids.to(torch.long))
        mask = prompt_mask.to(torch.float32).unsqueeze(-1)
        pooled = (emb * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return self.proj(pooled)


@dataclass(frozen=True)
class FacialExpressionModelConfig:
    vocab_size: int
    pad_id: int
    feature_dim: int = 16
    hidden_dim: int = 48
    text_dim: int = 24
    num_classes: int = 4


class CompactFacialExpressionVLM(nn.Module):
    def __init__(self, cfg: FacialExpressionModelConfig) -> None:
        super().__init__()
        self.face_encoder = nn.Sequential(
            nn.Linear(int(cfg.feature_dim), int(cfg.hidden_dim)),
            nn.ReLU(),
            nn.LayerNorm(int(cfg.hidden_dim)),
        )
        self.prompt_encoder = PromptEncoder(
            vocab_size=int(cfg.vocab_size),
            pad_id=int(cfg.pad_id),
            text_dim=int(cfg.text_dim),
            hidden_dim=int(cfg.hidden_dim),
        )
        self.fusion = nn.Sequential(
            nn.Linear(int(cfg.hidden_dim) * 2, int(cfg.hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(cfg.hidden_dim), int(cfg.hidden_dim)),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(int(cfg.hidden_dim), int(cfg.num_classes))

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        face_feat = self.face_encoder(batch["face_features"].to(torch.float32))
        prompt_feat = self.prompt_encoder(batch["prompt_ids"], batch["prompt_mask"])
        fused = self.fusion(torch.cat([face_feat, prompt_feat], dim=-1))
        logits = self.classifier(fused)
        probs = torch.softmax(logits, dim=-1)
        pred_labels = logits.argmax(dim=-1)
        return {"logits": logits, "probs": probs, "pred_labels": pred_labels}


def expression_loss(*, logits: torch.Tensor, labels: torch.Tensor) -> dict[str, torch.Tensor]:
    cls_loss = F.cross_entropy(logits, labels.to(torch.long))
    return {"loss": cls_loss, "cls_loss": cls_loss}


@torch.no_grad()
def classification_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    pred = logits.argmax(dim=-1)
    target = labels.to(torch.long)
    return float((pred == target).to(torch.float32).mean().item())


__all__ = [
    "FacialExpressionModelConfig",
    "PromptEncoder",
    "CompactFacialExpressionVLM",
    "classification_accuracy",
    "expression_loss",
]
