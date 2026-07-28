from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


class VisionEncoder(nn.Module):
    def __init__(self, *, vision_width: int, hidden_dim: int, grid_size: int) -> None:
        super().__init__()
        mid = max(16, int(vision_width) // 2)
        self.features = nn.Sequential(
            nn.Conv2d(3, mid, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(mid, int(vision_width), kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((int(grid_size), int(grid_size))),
        )
        self.proj = nn.Conv2d(int(vision_width), int(hidden_dim), kernel_size=1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        feat = self.features(image.to(torch.float32))
        return self.proj(feat)


class TextEncoder(nn.Module):
    def __init__(self, *, vocab_size: int, pad_id: int, text_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(int(vocab_size), int(text_dim), padding_idx=int(pad_id))
        self.proj = nn.Linear(int(text_dim), int(hidden_dim))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(input_ids.to(torch.long))
        mask = attention_mask.to(torch.float32).unsqueeze(-1)
        pooled = (emb * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return self.proj(pooled)


@dataclass(frozen=True)
class GroundingModelConfig:
    vocab_size: int
    pad_id: int
    image_size: int
    grid_size: int
    hidden_dim: int = 64
    vision_width: int = 32
    text_dim: int = 32


@dataclass(frozen=True)
class GroundingLossConfig:
    box_weight: float = 2.0


def _decode_boxes(
    *,
    cell_logits: torch.Tensor,
    box_deltas: torch.Tensor,
    grid_size: int,
) -> torch.Tensor:
    batch_size = int(cell_logits.shape[0])
    best_cell = cell_logits.argmax(dim=1)
    chosen = box_deltas[torch.arange(batch_size, device=cell_logits.device), best_cell]

    row = (best_cell // int(grid_size)).to(torch.float32)
    col = (best_cell % int(grid_size)).to(torch.float32)
    cell_size = 1.0 / float(grid_size)

    cx = (col + chosen[:, 0]) * cell_size
    cy = (row + chosen[:, 1]) * cell_size
    w = chosen[:, 2].clamp(1e-3, 1.0)
    h = chosen[:, 3].clamp(1e-3, 1.0)

    x1 = (cx - 0.5 * w).clamp(0.0, 1.0)
    y1 = (cy - 0.5 * h).clamp(0.0, 1.0)
    x2 = (cx + 0.5 * w).clamp(0.0, 1.0)
    y2 = (cy + 0.5 * h).clamp(0.0, 1.0)
    return torch.stack([x1, y1, x2, y2], dim=1)


class CompactGroundingModel(nn.Module):
    def __init__(self, cfg: GroundingModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.vision_encoder = VisionEncoder(
            vision_width=int(cfg.vision_width),
            hidden_dim=int(cfg.hidden_dim),
            grid_size=int(cfg.grid_size),
        )
        self.text_encoder = TextEncoder(
            vocab_size=int(cfg.vocab_size),
            pad_id=int(cfg.pad_id),
            text_dim=int(cfg.text_dim),
            hidden_dim=int(cfg.hidden_dim),
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(int(cfg.hidden_dim) * 2, int(cfg.hidden_dim), kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(int(cfg.hidden_dim), int(cfg.hidden_dim), kernel_size=1),
            nn.ReLU(),
        )
        self.cell_head = nn.Conv2d(int(cfg.hidden_dim), 1, kernel_size=1)
        self.box_head = nn.Conv2d(int(cfg.hidden_dim), 4, kernel_size=1)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        visual = self.vision_encoder(batch["image"])
        text = self.text_encoder(batch["input_ids"], batch["attention_mask"])
        text_map = text.view(int(text.shape[0]), int(text.shape[1]), 1, 1).expand_as(visual)
        fused = self.fusion(torch.cat([visual, text_map], dim=1))

        cell_logits = self.cell_head(fused).flatten(start_dim=1)
        box_deltas = torch.sigmoid(self.box_head(fused))
        box_deltas = box_deltas.flatten(start_dim=2).transpose(1, 2).contiguous()
        pred_boxes = _decode_boxes(
            cell_logits=cell_logits,
            box_deltas=box_deltas,
            grid_size=int(self.cfg.grid_size),
        )
        return {
            "cell_logits": cell_logits,
            "box_deltas": box_deltas,
            "pred_boxes": pred_boxes,
        }


def grounding_loss(
    *,
    cell_logits: torch.Tensor,
    box_deltas: torch.Tensor,
    target_cell: torch.Tensor,
    target_delta: torch.Tensor,
    cfg: GroundingLossConfig,
) -> dict[str, torch.Tensor]:
    cell_loss = F.cross_entropy(cell_logits, target_cell.to(torch.long))
    chosen = box_deltas[torch.arange(int(box_deltas.shape[0]), device=box_deltas.device), target_cell]
    box_loss = F.smooth_l1_loss(chosen, target_delta.to(torch.float32))
    total = cell_loss + float(cfg.box_weight) * box_loss
    return {"loss": total, "cell_loss": cell_loss, "box_loss": box_loss}


@torch.no_grad()
def bbox_l1_metric(pred_boxes: torch.Tensor, target_box: torch.Tensor) -> float:
    return float((pred_boxes - target_box).abs().mean().item())


@torch.no_grad()
def center_accuracy(pred_boxes: torch.Tensor, target_box: torch.Tensor) -> float:
    cx = 0.5 * (pred_boxes[:, 0] + pred_boxes[:, 2])
    cy = 0.5 * (pred_boxes[:, 1] + pred_boxes[:, 3])
    inside = (cx >= target_box[:, 0]) & (cx <= target_box[:, 2])
    inside = inside & (cy >= target_box[:, 1]) & (cy <= target_box[:, 3])
    return float(inside.to(torch.float32).mean().item())


__all__ = [
    "GroundingLossConfig",
    "GroundingModelConfig",
    "CompactGroundingModel",
    "VisionEncoder",
    "TextEncoder",
    "bbox_l1_metric",
    "center_accuracy",
    "grounding_loss",
]
