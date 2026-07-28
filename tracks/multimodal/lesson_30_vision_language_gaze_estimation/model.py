from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


class TinyVisionEncoder(nn.Module):
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

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4:
            raise ValueError(f"Expected image shape (B, C, H, W), got {tuple(image.shape)}")
        return self.net(image.to(torch.float32))


class MaskedTextEncoder(nn.Module):
    def __init__(self, *, vocab_size: int, pad_id: int, text_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(int(vocab_size), int(text_dim), padding_idx=int(pad_id))
        self.proj = nn.Linear(int(text_dim), int(text_dim))

    def forward(self, token_ids: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
        token_embed = self.embedding(token_ids.to(torch.long))
        mask = token_mask.to(torch.float32).unsqueeze(-1)
        pooled = (token_embed * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return self.proj(pooled)


@dataclass(frozen=True)
class GazeEstimationConfig:
    vocab_size: int
    pad_id: int
    image_size: int
    max_text_length: int
    heatmap_size: int
    hidden_dim: int = 64
    text_dim: int = 32
    vision_width: int = 32


class CompactVisionLanguageGazeEstimator(nn.Module):
    def __init__(self, cfg: GazeEstimationConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.vision_encoder = TinyVisionEncoder(vision_width=int(cfg.vision_width))
        self.text_encoder = MaskedTextEncoder(
            vocab_size=int(cfg.vocab_size),
            pad_id=int(cfg.pad_id),
            text_dim=int(cfg.text_dim),
        )
        self.head_proj = nn.Linear(2, int(cfg.text_dim))
        fused_dim = int(cfg.vision_width) + int(cfg.text_dim) * 2
        self.trunk = nn.Sequential(
            nn.Linear(fused_dim, int(cfg.hidden_dim)),
            nn.ReLU(),
        )
        self.point_head = nn.Linear(int(cfg.hidden_dim), 2)
        self.heatmap_head = nn.Linear(int(cfg.hidden_dim), int(cfg.heatmap_size * cfg.heatmap_size))

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        image_features = self.vision_encoder(batch["image"])
        text_features = self.text_encoder(batch["input_ids"], batch["attention_mask"])
        head_features = self.head_proj(batch["head_xy"].to(torch.float32))
        fused = torch.cat([image_features, text_features, head_features], dim=-1)
        hidden = self.trunk(fused)
        gaze_point = torch.sigmoid(self.point_head(hidden))
        gaze_heatmap = self.heatmap_head(hidden).view(
            int(hidden.shape[0]), 1, int(self.cfg.heatmap_size), int(self.cfg.heatmap_size)
        )
        gaze_heatmap = torch.sigmoid(gaze_heatmap)
        return {"gaze_point": gaze_point, "gaze_heatmap": gaze_heatmap}


def gaze_point_loss(pred_point: torch.Tensor, target_point: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred_point, target_point.to(torch.float32))


def gaze_heatmap_loss(pred_heatmap: torch.Tensor, target_heatmap: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred_heatmap, target_heatmap.to(torch.float32))


@torch.no_grad()
def gaze_point_l1(pred_point: torch.Tensor, target_point: torch.Tensor) -> float:
    return float((pred_point - target_point.to(torch.float32)).abs().mean().item())


__all__ = [
    "GazeEstimationConfig",
    "MaskedTextEncoder",
    "TinyVisionEncoder",
    "CompactVisionLanguageGazeEstimator",
    "gaze_heatmap_loss",
    "gaze_point_l1",
    "gaze_point_loss",
]
