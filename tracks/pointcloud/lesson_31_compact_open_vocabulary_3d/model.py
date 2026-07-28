from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    pad_id: int
    text_dim: int = 24
    point_dim: int = 48
    hidden_dim: int = 48
    num_classes: int = 3


class CompactOpenVocabulary3DModel(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        if int(cfg.vocab_size) <= 1:
            raise ValueError("vocab_size must be > 1")
        if int(cfg.num_classes) < 2:
            raise ValueError("num_classes must be >= 2")

        self.token_embed = nn.Embedding(
            num_embeddings=int(cfg.vocab_size),
            embedding_dim=int(cfg.text_dim),
            padding_idx=int(cfg.pad_id),
        )
        self.point_encoder = nn.Sequential(
            nn.Linear(3, int(cfg.point_dim)),
            nn.ReLU(),
            nn.Linear(int(cfg.point_dim), int(cfg.point_dim)),
            nn.ReLU(),
        )
        fused_global = int(cfg.point_dim) + int(cfg.text_dim)
        self.class_head = nn.Sequential(
            nn.Linear(fused_global, int(cfg.hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(cfg.hidden_dim), int(cfg.num_classes)),
        )
        self.mask_head = nn.Sequential(
            nn.Linear(fused_global, int(cfg.hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(cfg.hidden_dim), 1),
        )

    def forward(
        self,
        points: torch.Tensor,
        query_ids: torch.Tensor,
        query_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if points.ndim != 3 or points.size(-1) != 3:
            raise ValueError("points must have shape [batch, num_points, 3]")
        if query_ids.ndim != 2:
            raise ValueError("query_ids must have shape [batch, seq_len]")
        if query_mask.shape != query_ids.shape:
            raise ValueError("query_mask and query_ids must have the same shape")
        if query_ids.size(0) != points.size(0):
            raise ValueError("points and query tensors must share the batch size")

        batch_size, num_points, _ = points.shape
        token_feat = self.token_embed(query_ids)
        norm = query_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        text_feat = (token_feat * query_mask.unsqueeze(-1)).sum(dim=1) / norm

        point_feat = self.point_encoder(points.reshape(batch_size * num_points, 3)).reshape(
            batch_size, num_points, -1
        )
        global_feat = point_feat.mean(dim=1)
        global_fused = torch.cat((global_feat, text_feat), dim=-1)
        class_logits = self.class_head(global_fused)

        text_tiled = text_feat.unsqueeze(1).expand(-1, num_points, -1)
        point_fused = torch.cat((point_feat, text_tiled), dim=-1)
        mask_logits = self.mask_head(point_fused).squeeze(-1)
        return {"class_logits": class_logits, "mask_logits": mask_logits}


def open_vocabulary_3d_loss(
    class_logits: torch.Tensor,
    mask_logits: torch.Tensor,
    class_targets: torch.Tensor,
    mask_targets: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    if class_logits.ndim != 2:
        raise ValueError("class_logits must have shape [batch, num_classes]")
    if mask_logits.ndim != 2:
        raise ValueError("mask_logits must have shape [batch, num_points]")
    if class_targets.ndim != 1:
        raise ValueError("class_targets must have shape [batch]")
    if mask_targets.shape != mask_logits.shape:
        raise ValueError("mask_targets and mask_logits must have the same shape")

    class_loss = F.cross_entropy(class_logits, class_targets)
    mask_loss = F.binary_cross_entropy_with_logits(mask_logits, mask_targets)
    total = class_loss + mask_loss
    return total, {
        "class_loss": float(class_loss.detach().item()),
        "mask_loss": float(mask_loss.detach().item()),
    }


def mask_iou(mask_logits: torch.Tensor, mask_targets: torch.Tensor) -> float:
    if mask_logits.shape != mask_targets.shape:
        raise ValueError("mask_logits and mask_targets must have the same shape")

    pred = torch.sigmoid(mask_logits) >= 0.5
    target = mask_targets >= 0.5
    inter = (pred & target).sum(dim=1).to(torch.float32)
    union = (pred | target).sum(dim=1).to(torch.float32).clamp(min=1.0)
    return float((inter / union).mean().item())


__all__ = ["ModelConfig", "CompactOpenVocabulary3DModel", "mask_iou", "open_vocabulary_3d_loss"]
