from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


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


class ScaleMapHead(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.cell_projector = nn.Sequential(
            nn.Linear(int(hidden_dim) * 4, int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.ReLU(),
        )
        self.map_conv = nn.Sequential(
            nn.Conv2d(int(hidden_dim), int(hidden_dim), kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(int(hidden_dim), int(hidden_dim), kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.score_head = nn.Conv2d(int(hidden_dim), 1, kernel_size=1)

    def _build_map_features(self, temporal_features: torch.Tensor, query_features: torch.Tensor) -> torch.Tensor:
        batch_size, num_frames, hidden_dim = temporal_features.shape
        map_features = temporal_features.new_zeros((batch_size, num_frames, num_frames, hidden_dim))
        cumsum = torch.cat(
            [temporal_features.new_zeros((batch_size, 1, hidden_dim)), temporal_features.cumsum(dim=1)],
            dim=1,
        )

        for start_idx in range(num_frames):
            start_feat = temporal_features[:, start_idx]
            for end_idx in range(start_idx, num_frames):
                end_feat = temporal_features[:, end_idx]
                seg_sum = cumsum[:, end_idx + 1] - cumsum[:, start_idx]
                seg_mean = seg_sum / float(end_idx - start_idx + 1)
                cell_input = torch.cat([start_feat, end_feat, seg_mean, query_features], dim=-1)
                map_features[:, start_idx, end_idx] = self.cell_projector(cell_input)
        return map_features

    def forward(self, temporal_features: torch.Tensor, query_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        map_features = self._build_map_features(temporal_features, query_features)
        map_tensor = map_features.permute(0, 3, 1, 2).contiguous()
        refined = self.map_conv(map_tensor)
        score_map = self.score_head(refined).squeeze(1)
        return score_map, map_features


@dataclass(frozen=True)
class MultiScaleTwoDtanModelConfig:
    vocab_size: int
    pad_id: int
    num_frames: int
    hidden_dim: int = 64
    vision_width: int = 32
    text_dim: int = 32


class ToyMultiScaleTwoDtanTemporalGroundingModel(nn.Module):
    def __init__(self, cfg: MultiScaleTwoDtanModelConfig) -> None:
        super().__init__()
        if int(cfg.num_frames) % 4 != 0:
            raise ValueError("num_frames must be divisible by 4")
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
        self.scale1_head = ScaleMapHead(int(cfg.hidden_dim))
        self.scale2_head = ScaleMapHead(int(cfg.hidden_dim))
        self.scale3_head = ScaleMapHead(int(cfg.hidden_dim))
        self.fusion_head = nn.Sequential(
            nn.Conv2d(3, int(cfg.hidden_dim), kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(int(cfg.hidden_dim), 1, kernel_size=1),
        )
        self.register_buffer(
            "map_mask_s1_template",
            torch.triu(torch.ones(int(cfg.num_frames), int(cfg.num_frames), dtype=torch.float32)),
            persistent=False,
        )
        self.register_buffer(
            "map_mask_s2_template",
            torch.triu(torch.ones(int(cfg.num_frames) // 2, int(cfg.num_frames) // 2, dtype=torch.float32)),
            persistent=False,
        )
        self.register_buffer(
            "map_mask_s3_template",
            torch.triu(torch.ones(int(cfg.num_frames) // 4, int(cfg.num_frames) // 4, dtype=torch.float32)),
            persistent=False,
        )

    def _downsample_temporal(self, temporal_features: torch.Tensor, factor: int) -> torch.Tensor:
        if int(factor) == 1:
            return temporal_features
        batch_size, num_frames, hidden_dim = temporal_features.shape
        scale_len = int(num_frames) // int(factor)
        pooled = temporal_features[:, : scale_len * int(factor)]
        pooled = pooled.view(batch_size, scale_len, int(factor), hidden_dim)
        return pooled.mean(dim=2)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        frame_features = self.frame_encoder(batch["video"])
        temporal_features = self.temporal_encoder(frame_features)
        query_features = self.query_encoder(batch["query_ids"], batch["attention_mask"])

        temporal_s1 = self._downsample_temporal(temporal_features, factor=1)
        temporal_s2 = self._downsample_temporal(temporal_features, factor=2)
        temporal_s3 = self._downsample_temporal(temporal_features, factor=4)

        score_map_s1, map_features_s1 = self.scale1_head(temporal_s1, query_features)
        score_map_s2, map_features_s2 = self.scale2_head(temporal_s2, query_features)
        score_map_s3, map_features_s3 = self.scale3_head(temporal_s3, query_features)

        target_size = (int(score_map_s1.shape[-2]), int(score_map_s1.shape[-1]))
        score_map_s2_up = F.interpolate(
            score_map_s2.unsqueeze(1),
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )
        score_map_s3_up = F.interpolate(
            score_map_s3.unsqueeze(1),
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )
        fused_input = torch.cat([score_map_s1.unsqueeze(1), score_map_s2_up, score_map_s3_up], dim=1)
        fused_score_map = self.fusion_head(fused_input).squeeze(1)

        pred_segment_s1 = decode_best_segments(
            score_map=score_map_s1,
            map_mask=self.map_mask_s1_template.unsqueeze(0).expand(int(score_map_s1.shape[0]), -1, -1),
            stride=1,
            max_frame_index=int(self.cfg.num_frames) - 1,
        )
        pred_segment_s2 = decode_best_segments(
            score_map=score_map_s2,
            map_mask=self.map_mask_s2_template.unsqueeze(0).expand(int(score_map_s2.shape[0]), -1, -1),
            stride=2,
            max_frame_index=int(self.cfg.num_frames) - 1,
        )
        pred_segment_s3 = decode_best_segments(
            score_map=score_map_s3,
            map_mask=self.map_mask_s3_template.unsqueeze(0).expand(int(score_map_s3.shape[0]), -1, -1),
            stride=4,
            max_frame_index=int(self.cfg.num_frames) - 1,
        )
        pred_segment_fused = decode_best_segments(
            score_map=fused_score_map,
            map_mask=self.map_mask_s1_template.unsqueeze(0).expand(int(fused_score_map.shape[0]), -1, -1),
            stride=1,
            max_frame_index=int(self.cfg.num_frames) - 1,
        )

        return {
            "score_map_s1": score_map_s1,
            "score_map_s2": score_map_s2,
            "score_map_s3": score_map_s3,
            "fused_score_map": fused_score_map,
            "map_features_s1": map_features_s1,
            "map_features_s2": map_features_s2,
            "map_features_s3": map_features_s3,
            "pred_segment_s1": pred_segment_s1,
            "pred_segment_s2": pred_segment_s2,
            "pred_segment_s3": pred_segment_s3,
            "pred_segment_fused": pred_segment_fused,
            "pred_segments": pred_segment_fused,
        }


def _masked_map_mse(
    *,
    score_map: torch.Tensor,
    map_labels: torch.Tensor,
    map_mask: torch.Tensor,
) -> torch.Tensor:
    error = (torch.sigmoid(score_map) - map_labels.to(torch.float32)).pow(2)
    loss = (error * map_mask.to(torch.float32)).sum()
    return loss / map_mask.to(torch.float32).sum().clamp_min(1.0)


def multiscale_temporal_map_loss(
    *,
    score_map_s1: torch.Tensor,
    score_map_s2: torch.Tensor,
    score_map_s3: torch.Tensor,
    fused_score_map: torch.Tensor,
    map_labels_s1: torch.Tensor,
    map_mask_s1: torch.Tensor,
    map_labels_s2: torch.Tensor,
    map_mask_s2: torch.Tensor,
    map_labels_s3: torch.Tensor,
    map_mask_s3: torch.Tensor,
) -> dict[str, torch.Tensor]:
    aux_loss_s1 = _masked_map_mse(
        score_map=score_map_s1,
        map_labels=map_labels_s1,
        map_mask=map_mask_s1,
    )
    aux_loss_s2 = _masked_map_mse(
        score_map=score_map_s2,
        map_labels=map_labels_s2,
        map_mask=map_mask_s2,
    )
    aux_loss_s3 = _masked_map_mse(
        score_map=score_map_s3,
        map_labels=map_labels_s3,
        map_mask=map_mask_s3,
    )
    fused_loss = _masked_map_mse(
        score_map=fused_score_map,
        map_labels=map_labels_s1,
        map_mask=map_mask_s1,
    )
    loss = fused_loss + 0.5 * (aux_loss_s1 + aux_loss_s2 + aux_loss_s3) / 3.0
    return {
        "loss": loss,
        "fused_loss": fused_loss,
        "aux_loss_s1": aux_loss_s1,
        "aux_loss_s2": aux_loss_s2,
        "aux_loss_s3": aux_loss_s3,
    }


@torch.no_grad()
def decode_best_segments(
    *,
    score_map: torch.Tensor,
    map_mask: torch.Tensor,
    stride: int,
    max_frame_index: int,
) -> torch.Tensor:
    masked = torch.where(map_mask > 0.5, score_map, torch.full_like(score_map, -1.0e9))
    batch_size, num_frames, _num_frames = masked.shape
    flat_idx = masked.view(batch_size, -1).argmax(dim=1)
    start_idx = flat_idx // int(num_frames)
    end_idx = flat_idx % int(num_frames)
    start_frame = start_idx * int(stride)
    end_frame = (end_idx + 1) * int(stride) - 1
    end_frame = end_frame.clamp_max(int(max_frame_index))
    return torch.stack([start_frame, end_frame], dim=1).to(torch.long)


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
    "MultiScaleTwoDtanModelConfig",
    "QueryEncoder",
    "ScaleMapHead",
    "TemporalVideoEncoder",
    "TinyFrameEncoder",
    "ToyMultiScaleTwoDtanTemporalGroundingModel",
    "decode_best_segments",
    "multiscale_temporal_map_loss",
    "recall_at_iou",
    "temporal_iou_metric",
]
