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

    def forward(self, query_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(query_ids.to(torch.long))
        mask = attention_mask.to(torch.float32).unsqueeze(-1)
        pooled = (emb * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return self.proj(pooled)


@dataclass(frozen=True)
class OwlVitModelConfig:
    vocab_size: int
    pad_id: int
    image_size: int
    grid_size: int
    hidden_dim: int = 64
    vision_width: int = 32
    text_dim: int = 32


@dataclass(frozen=True)
class OwlVitLossConfig:
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


class ToyOwlVitModel(nn.Module):
    def __init__(self, cfg: OwlVitModelConfig) -> None:
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
        self.presence_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(int(cfg.hidden_dim), 1),
        )
        self.cell_head = nn.Conv2d(int(cfg.hidden_dim), 1, kernel_size=1)
        self.box_head = nn.Conv2d(int(cfg.hidden_dim), 4, kernel_size=1)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        visual = self.vision_encoder(batch["image"])
        text = self.text_encoder(batch["query_ids"], batch["attention_mask"])
        text_map = text.view(int(text.shape[0]), int(text.shape[1]), 1, 1).expand_as(visual)
        fused = self.fusion(torch.cat([visual, text_map], dim=1))

        presence_logit = self.presence_head(fused).squeeze(-1)
        cell_logits = self.cell_head(fused).flatten(start_dim=1)
        box_deltas = torch.sigmoid(self.box_head(fused))
        box_deltas = box_deltas.flatten(start_dim=2).transpose(1, 2).contiguous()
        pred_boxes = _decode_boxes(
            cell_logits=cell_logits,
            box_deltas=box_deltas,
            grid_size=int(self.cfg.grid_size),
        )
        return {
            "presence_logit": presence_logit,
            "cell_logits": cell_logits,
            "box_deltas": box_deltas,
            "pred_boxes": pred_boxes,
        }


def owlvit_loss(
    *,
    presence_logit: torch.Tensor,
    cell_logits: torch.Tensor,
    box_deltas: torch.Tensor,
    target_present: torch.Tensor,
    target_cell: torch.Tensor,
    target_delta: torch.Tensor,
    cfg: OwlVitLossConfig,
) -> dict[str, torch.Tensor]:
    target_present = target_present.to(torch.float32)
    presence_loss = F.binary_cross_entropy_with_logits(presence_logit, target_present)

    positive_mask = target_present > 0.5
    if bool(positive_mask.any()):
        pos_cell_logits = cell_logits[positive_mask]
        pos_target_cell = target_cell[positive_mask].to(torch.long)
        cell_loss = F.cross_entropy(pos_cell_logits, pos_target_cell)
        chosen = box_deltas[positive_mask][
            torch.arange(int(pos_target_cell.shape[0]), device=box_deltas.device),
            pos_target_cell,
        ]
        box_loss = F.smooth_l1_loss(chosen, target_delta[positive_mask].to(torch.float32))
    else:
        zero = presence_loss.new_zeros(())
        cell_loss = zero
        box_loss = zero

    total = presence_loss + cell_loss + float(cfg.box_weight) * box_loss
    return {
        "loss": total,
        "presence_loss": presence_loss,
        "cell_loss": cell_loss,
        "box_loss": box_loss,
    }


@torch.no_grad()
def presence_accuracy(presence_logit: torch.Tensor, target_present: torch.Tensor) -> float:
    pred = (torch.sigmoid(presence_logit) >= 0.5).to(torch.float32)
    target = target_present.to(torch.float32)
    return float((pred == target).to(torch.float32).mean().item())


@torch.no_grad()
def bbox_l1_metric(pred_boxes: torch.Tensor, target_box: torch.Tensor, target_present: torch.Tensor) -> float:
    positive_mask = target_present > 0.5
    if not bool(positive_mask.any()):
        return 0.0
    return float((pred_boxes[positive_mask] - target_box[positive_mask]).abs().mean().item())


@torch.no_grad()
def center_accuracy(pred_boxes: torch.Tensor, target_box: torch.Tensor, target_present: torch.Tensor) -> float:
    positive_mask = target_present > 0.5
    if not bool(positive_mask.any()):
        return 0.0
    pred = pred_boxes[positive_mask]
    target = target_box[positive_mask]
    cx = 0.5 * (pred[:, 0] + pred[:, 2])
    cy = 0.5 * (pred[:, 1] + pred[:, 3])
    inside = (cx >= target[:, 0]) & (cx <= target[:, 2])
    inside = inside & (cy >= target[:, 1]) & (cy <= target[:, 3])
    return float(inside.to(torch.float32).mean().item())


__all__ = [
    "OwlVitLossConfig",
    "OwlVitModelConfig",
    "TextEncoder",
    "ToyOwlVitModel",
    "VisionEncoder",
    "bbox_l1_metric",
    "center_accuracy",
    "owlvit_loss",
    "presence_accuracy",
]
