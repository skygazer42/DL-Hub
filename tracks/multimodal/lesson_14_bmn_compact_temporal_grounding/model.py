from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


class TinyFrameEncoder(nn.Module):
    def __init__(self, vision_width: int) -> None:
        super().__init__()
        mid = max(16, int(vision_width) // 2)
        self.features = nn.Sequential(
            nn.Conv2d(3, mid, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(mid, int(vision_width), kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(int(vision_width), int(vision_width), kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        batch_size, num_frames, channels, height, width = video.shape
        frames = video.view(batch_size * num_frames, channels, height, width).to(torch.float32)
        feat = self.features(frames).flatten(start_dim=1)
        return feat.view(batch_size, num_frames, -1)


class TemporalVideoEncoder(nn.Module):
    def __init__(self, *, in_dim: int, hidden_dim: int, num_frames: int) -> None:
        super().__init__()
        self.input_proj = nn.Linear(int(in_dim), int(hidden_dim))
        self.pos_embed = nn.Parameter(torch.zeros(int(num_frames), int(hidden_dim)))
        self.temporal_rnn = nn.GRU(int(hidden_dim), int(hidden_dim), batch_first=True)
        self.norm = nn.LayerNorm(int(hidden_dim))

    def forward(self, frame_features: torch.Tensor) -> torch.Tensor:
        feat = self.input_proj(frame_features)
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


class QueryConditionedFusion(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(int(hidden_dim) * 2, int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.ReLU(),
        )

    def forward(self, video_features: torch.Tensor, query_features: torch.Tensor) -> torch.Tensor:
        query_map = query_features.unsqueeze(1).expand(-1, int(video_features.shape[1]), -1)
        return self.fusion(torch.cat([video_features, query_map], dim=-1))


@dataclass(frozen=True)
class BmnModelConfig:
    vocab_size: int
    pad_id: int
    num_frames: int
    hidden_dim: int = 64
    vision_width: int = 32
    text_dim: int = 32


class CompactBmnTemporalGroundingModel(nn.Module):
    def __init__(self, cfg: BmnModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.frame_encoder = TinyFrameEncoder(int(cfg.vision_width))
        self.temporal_encoder = TemporalVideoEncoder(
            in_dim=int(cfg.vision_width),
            hidden_dim=int(cfg.hidden_dim),
            num_frames=int(cfg.num_frames),
        )
        self.query_encoder = QueryEncoder(
            vocab_size=int(cfg.vocab_size),
            pad_id=int(cfg.pad_id),
            text_dim=int(cfg.text_dim),
            hidden_dim=int(cfg.hidden_dim),
        )
        self.fusion = QueryConditionedFusion(int(cfg.hidden_dim))
        self.start_head = nn.Linear(int(cfg.hidden_dim), 1)
        self.end_head = nn.Linear(int(cfg.hidden_dim), 1)
        self.proposal_head = nn.Sequential(
            nn.Linear(int(cfg.hidden_dim) * 3, int(cfg.hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(cfg.hidden_dim), 1),
        )
        self.register_buffer(
            "proposal_mask_template",
            torch.triu(torch.ones(int(cfg.num_frames), int(cfg.num_frames), dtype=torch.float32)),
            persistent=False,
        )

    def _proposal_features(self, fused: torch.Tensor) -> torch.Tensor:
        batch_size, num_frames, hidden_dim = fused.shape
        proposal_scores = fused.new_zeros((batch_size, num_frames, num_frames))
        cumsum = torch.cat(
            [fused.new_zeros((batch_size, 1, hidden_dim)), fused.cumsum(dim=1)],
            dim=1,
        )

        for start_idx in range(num_frames):
            start_feat = fused[:, start_idx]
            for end_idx in range(start_idx, num_frames):
                end_feat = fused[:, end_idx]
                seg_sum = cumsum[:, end_idx + 1] - cumsum[:, start_idx]
                seg_mean = seg_sum / float(end_idx - start_idx + 1)
                proposal_feat = torch.cat([start_feat, end_feat, seg_mean], dim=-1)
                proposal_scores[:, start_idx, end_idx] = self.proposal_head(proposal_feat).squeeze(-1)

        return proposal_scores

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        frame_features = self.frame_encoder(batch["video"])
        video_features = self.temporal_encoder(frame_features)
        query_features = self.query_encoder(batch["query_ids"], batch["attention_mask"])
        fused = self.fusion(video_features, query_features)
        start_logits = self.start_head(fused).squeeze(-1)
        end_logits = self.end_head(fused).squeeze(-1)
        proposal_scores = self._proposal_features(fused)
        pred_segments = decode_best_segments(
            start_logits=start_logits,
            end_logits=end_logits,
            proposal_scores=proposal_scores,
            proposal_mask=self.proposal_mask_template.unsqueeze(0).expand(int(fused.shape[0]), -1, -1),
        )
        return {
            "start_logits": start_logits,
            "end_logits": end_logits,
            "proposal_scores": proposal_scores,
            "pred_segments": pred_segments,
        }


def temporal_grounding_loss(
    *,
    start_logits: torch.Tensor,
    end_logits: torch.Tensor,
    proposal_scores: torch.Tensor,
    start_labels: torch.Tensor,
    end_labels: torch.Tensor,
    proposal_labels: torch.Tensor,
    proposal_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    start_loss = F.binary_cross_entropy_with_logits(
        start_logits, start_labels.to(torch.float32)
    )
    end_loss = F.binary_cross_entropy_with_logits(
        end_logits, end_labels.to(torch.float32)
    )
    proposal_error = (torch.sigmoid(proposal_scores) - proposal_labels.to(torch.float32)).pow(2)
    proposal_loss = (proposal_error * proposal_mask.to(torch.float32)).sum()
    proposal_loss = proposal_loss / proposal_mask.to(torch.float32).sum().clamp_min(1.0)
    total = start_loss + end_loss + proposal_loss
    return {
        "loss": total,
        "start_loss": start_loss,
        "end_loss": end_loss,
        "proposal_loss": proposal_loss,
    }


@torch.no_grad()
def decode_best_segments(
    *,
    start_logits: torch.Tensor,
    end_logits: torch.Tensor,
    proposal_scores: torch.Tensor,
    proposal_mask: torch.Tensor,
) -> torch.Tensor:
    start_prob = torch.sigmoid(start_logits).unsqueeze(-1)
    end_prob = torch.sigmoid(end_logits).unsqueeze(1)
    proposal_prob = torch.sigmoid(proposal_scores)
    score_map = start_prob * end_prob * proposal_prob
    score_map = torch.where(proposal_mask > 0.5, score_map, torch.full_like(score_map, -1.0))

    batch_size, num_frames, _num_frames = score_map.shape
    flat_idx = score_map.view(batch_size, -1).argmax(dim=1)
    start_idx = flat_idx // int(num_frames)
    end_idx = flat_idx % int(num_frames)
    return torch.stack([start_idx, end_idx], dim=1).to(torch.long)


@torch.no_grad()
def start_accuracy(start_logits: torch.Tensor, target_segment: torch.Tensor) -> float:
    pred = start_logits.argmax(dim=1)
    target = target_segment[:, 0].to(torch.long)
    return float((pred == target).to(torch.float32).mean().item())


@torch.no_grad()
def end_accuracy(end_logits: torch.Tensor, target_segment: torch.Tensor) -> float:
    pred = end_logits.argmax(dim=1)
    target = target_segment[:, 1].to(torch.long)
    return float((pred == target).to(torch.float32).mean().item())


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
    "BmnModelConfig",
    "QueryConditionedFusion",
    "QueryEncoder",
    "TemporalVideoEncoder",
    "TinyFrameEncoder",
    "CompactBmnTemporalGroundingModel",
    "decode_best_segments",
    "end_accuracy",
    "recall_at_iou",
    "start_accuracy",
    "temporal_grounding_loss",
    "temporal_iou_metric",
]
