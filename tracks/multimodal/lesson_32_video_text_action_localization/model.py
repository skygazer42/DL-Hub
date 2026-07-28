from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


class TemporalVideoEncoder(nn.Module):
    def __init__(self, *, feature_dim: int, hidden_dim: int, num_frames: int) -> None:
        super().__init__()
        self.input_proj = nn.Linear(int(feature_dim), int(hidden_dim))
        self.pos_embed = nn.Parameter(torch.zeros(int(num_frames), int(hidden_dim)))
        self.temporal_rnn = nn.GRU(int(hidden_dim), int(hidden_dim), batch_first=True)
        self.norm = nn.LayerNorm(int(hidden_dim))

    def forward(self, video_features: torch.Tensor) -> torch.Tensor:
        feat = self.input_proj(video_features.to(torch.float32))
        seq_len = int(feat.shape[1])
        feat = feat + self.pos_embed[:seq_len].unsqueeze(0)
        out, _ = self.temporal_rnn(feat)
        return self.norm(out)


class QueryEncoder(nn.Module):
    def __init__(self, *, vocab_size: int, pad_id: int, text_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(int(vocab_size), int(text_dim), padding_idx=int(pad_id))
        self.query_rnn = nn.GRU(int(text_dim), int(hidden_dim), batch_first=True)
        self.norm = nn.LayerNorm(int(hidden_dim))

    def forward(self, query_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        del attention_mask
        emb = self.embedding(query_ids.to(torch.long))
        _out, hidden = self.query_rnn(emb)
        return self.norm(hidden[-1])


@dataclass(frozen=True)
class ActionLocalizationModelConfig:
    vocab_size: int
    pad_id: int
    num_frames: int
    feature_dim: int = 24
    hidden_dim: int = 64
    text_dim: int = 32


class CompactActionLocalizationModel(nn.Module):
    def __init__(self, cfg: ActionLocalizationModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.video_encoder = TemporalVideoEncoder(
            feature_dim=int(cfg.feature_dim),
            hidden_dim=int(cfg.hidden_dim),
            num_frames=int(cfg.num_frames),
        )
        self.query_encoder = QueryEncoder(
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
        self.mask_head = nn.Linear(int(cfg.hidden_dim), 1)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        video_feat = self.video_encoder(batch["video_features"])
        query_feat = self.query_encoder(batch["query_ids"], batch["attention_mask"])
        query_map = query_feat.unsqueeze(1).expand(-1, int(video_feat.shape[1]), -1)
        fused = self.fusion(torch.cat([video_feat, query_map], dim=-1))
        mask_logits = self.mask_head(fused).squeeze(-1)
        pred_segments = decode_segments_from_mask(mask_logits)
        return {
            "mask_logits": mask_logits,
            "pred_segments": pred_segments,
        }


def action_localization_loss(*, mask_logits: torch.Tensor, action_mask: torch.Tensor) -> dict[str, torch.Tensor]:
    mask_loss = F.binary_cross_entropy_with_logits(mask_logits, action_mask.to(torch.float32))
    return {"loss": mask_loss, "mask_loss": mask_loss}


@torch.no_grad()
def decode_segments_from_mask(mask_logits: torch.Tensor, *, threshold: float = 0.5) -> torch.Tensor:
    probs = torch.sigmoid(mask_logits)
    binary = probs >= float(threshold)
    batch_size, num_frames = binary.shape
    segments = torch.zeros((batch_size, 2), dtype=torch.long, device=mask_logits.device)

    for idx in range(batch_size):
        active = torch.nonzero(binary[idx], as_tuple=False).flatten()
        if int(active.numel()) == 0:
            best = int(probs[idx].argmax().item())
            segments[idx, 0] = best
            segments[idx, 1] = best
            continue
        segments[idx, 0] = int(active[0].item())
        segments[idx, 1] = int(active[-1].item())

    return segments


@torch.no_grad()
def temporal_iou_metric(pred_segments: torch.Tensor, target_segment: torch.Tensor) -> float:
    pred_start = pred_segments[:, 0].to(torch.float32)
    pred_end = pred_segments[:, 1].to(torch.float32)
    target_start = target_segment[:, 0].to(torch.float32)
    target_end = target_segment[:, 1].to(torch.float32)

    inter = (torch.minimum(pred_end, target_end) - torch.maximum(pred_start, target_start) + 1.0).clamp_min(0.0)
    union = (pred_end - pred_start + 1.0) + (target_end - target_start + 1.0) - inter
    return float((inter / union.clamp_min(1.0)).mean().item())


@torch.no_grad()
def recall_at_iou(pred_segments: torch.Tensor, target_segment: torch.Tensor, *, threshold: float = 0.5) -> float:
    pred_start = pred_segments[:, 0].to(torch.float32)
    pred_end = pred_segments[:, 1].to(torch.float32)
    target_start = target_segment[:, 0].to(torch.float32)
    target_end = target_segment[:, 1].to(torch.float32)

    inter = (torch.minimum(pred_end, target_end) - torch.maximum(pred_start, target_start) + 1.0).clamp_min(0.0)
    union = (pred_end - pred_start + 1.0) + (target_end - target_start + 1.0) - inter
    tiou = inter / union.clamp_min(1.0)
    return float((tiou >= float(threshold)).to(torch.float32).mean().item())


__all__ = [
    "ActionLocalizationModelConfig",
    "QueryEncoder",
    "TemporalVideoEncoder",
    "CompactActionLocalizationModel",
    "action_localization_loss",
    "decode_segments_from_mask",
    "recall_at_iou",
    "temporal_iou_metric",
]
