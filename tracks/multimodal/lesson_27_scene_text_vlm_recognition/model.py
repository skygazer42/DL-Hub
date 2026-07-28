from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


class TinySceneEncoder(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        hidden = max(16, int(width) // 2)
        self.net = nn.Sequential(
            nn.Conv2d(3, hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden, int(width), kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4:
            raise ValueError(f"Expected image shape (B, C, H, W), got {tuple(image.shape)}")
        return self.net(image.to(torch.float32))


class TextPromptEncoder(nn.Module):
    def __init__(self, *, vocab_size: int, pad_id: int, width: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(int(vocab_size), int(width), padding_idx=int(pad_id))
        self.proj = nn.Linear(int(width), int(width))

    def forward(self, prompt_ids: torch.Tensor, prompt_mask: torch.Tensor) -> torch.Tensor:
        token_embed = self.embedding(prompt_ids.to(torch.long))
        mask = prompt_mask.to(torch.float32).unsqueeze(-1)
        pooled = (token_embed * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return self.proj(pooled)


@dataclass(frozen=True)
class SceneTextRecognizerConfig:
    vocab_size: int
    pad_id: int
    num_words: int = 4
    hidden_dim: int = 48
    vision_width: int = 32
    text_width: int = 32


class CompactSceneTextRecognizer(nn.Module):
    def __init__(self, cfg: SceneTextRecognizerConfig) -> None:
        super().__init__()
        self.scene_encoder = TinySceneEncoder(int(cfg.vision_width))
        self.prompt_encoder = TextPromptEncoder(
            vocab_size=int(cfg.vocab_size),
            pad_id=int(cfg.pad_id),
            width=int(cfg.text_width),
        )
        self.head = nn.Sequential(
            nn.Linear(int(cfg.vision_width + cfg.text_width), int(cfg.hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(cfg.hidden_dim), int(cfg.num_words)),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        scene_features = self.scene_encoder(batch["image"])
        prompt_features = self.prompt_encoder(batch["prompt_ids"], batch["prompt_mask"])
        fused = torch.cat([scene_features, prompt_features], dim=-1)
        logits = self.head(fused)
        probs = torch.softmax(logits, dim=-1)
        return {"logits": logits, "probs": probs}


def recognition_loss(logits: torch.Tensor, label_ids: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, label_ids.to(torch.long))


@torch.no_grad()
def recognition_accuracy(logits: torch.Tensor, label_ids: torch.Tensor) -> float:
    pred = logits.argmax(dim=-1)
    return float((pred == label_ids.to(torch.long)).to(torch.float32).mean().item())


__all__ = [
    "SceneTextRecognizerConfig",
    "TextPromptEncoder",
    "TinySceneEncoder",
    "CompactSceneTextRecognizer",
    "recognition_accuracy",
    "recognition_loss",
]
