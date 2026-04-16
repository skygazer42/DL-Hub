from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


class TinyObservationEncoder(nn.Module):
    def __init__(self, *, vision_width: int) -> None:
        super().__init__()
        hidden = max(16, int(vision_width) // 2)
        self.net = nn.Sequential(
            nn.Conv2d(3, hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden, int(vision_width), kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        if observation.ndim != 4:
            raise ValueError(f"Expected observation shape (B, C, H, W), got {tuple(observation.shape)}")
        return self.net(observation.to(torch.float32))


class VisionTextEncoder(nn.Module):
    def __init__(self, *, vocab_size: int, pad_id: int, text_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(int(vocab_size), int(text_dim), padding_idx=int(pad_id))
        self.proj = nn.Linear(int(text_dim), int(text_dim))

    def forward(self, instruction_ids: torch.Tensor, instruction_mask: torch.Tensor) -> torch.Tensor:
        embeds = self.embedding(instruction_ids.to(torch.long))
        mask = instruction_mask.to(torch.float32).unsqueeze(-1)
        pooled = (embeds * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return self.proj(pooled)


@dataclass(frozen=True)
class VisionLanguageNavigationConfig:
    vocab_size: int
    pad_id: int
    num_actions: int = 4
    hidden_dim: int = 64
    text_dim: int = 32
    vision_width: int = 32


class ToyVisionLanguageNavigationModel(nn.Module):
    def __init__(self, cfg: VisionLanguageNavigationConfig) -> None:
        super().__init__()
        self.obs_encoder = TinyObservationEncoder(vision_width=int(cfg.vision_width))
        self.text_encoder = VisionTextEncoder(
            vocab_size=int(cfg.vocab_size),
            pad_id=int(cfg.pad_id),
            text_dim=int(cfg.text_dim),
        )
        self.policy_head = nn.Sequential(
            nn.Linear(int(cfg.vision_width + cfg.text_dim), int(cfg.hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(cfg.hidden_dim), int(cfg.num_actions)),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        obs_features = self.obs_encoder(batch["observation"])
        text_features = self.text_encoder(batch["instruction_ids"], batch["instruction_mask"])
        fused = torch.cat([obs_features, text_features], dim=-1)
        logits = self.policy_head(fused)
        policy = torch.softmax(logits, dim=-1)
        return {"logits": logits, "policy": policy}


def navigation_loss(logits: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, actions.to(torch.long))


@torch.no_grad()
def navigation_accuracy(logits: torch.Tensor, actions: torch.Tensor) -> float:
    pred = logits.argmax(dim=-1)
    acc = (pred == actions.to(torch.long)).to(torch.float32).mean()
    return float(acc.item())


__all__ = [
    "TinyObservationEncoder",
    "ToyVisionLanguageNavigationModel",
    "VisionLanguageNavigationConfig",
    "VisionTextEncoder",
    "navigation_accuracy",
    "navigation_loss",
]
