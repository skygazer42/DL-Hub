from __future__ import annotations

import torch
from torch import nn

from dlhub.pointcloud.detection3d._common import PointNetEncoder


def check_sequence(points: torch.Tensor) -> None:
    if not isinstance(points, torch.Tensor):
        raise TypeError(f"points must be a torch.Tensor, got {type(points).__name__}")
    if points.ndim != 4:
        raise ValueError(f"points must have shape (B, T, N, C), got {tuple(points.shape)}")
    if points.shape[-1] < 3:
        raise ValueError(f"points last dim must be >=3 (xyz), got C={points.shape[-1]}")


class TemporalPointEncoder(nn.Module):
    def __init__(self, *, in_channels: int, width: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.point = PointNetEncoder(int(in_channels), width=int(width), dropout=float(dropout))
        self.temporal = nn.GRU(int(width), int(width), num_layers=1, batch_first=True)

    def forward(self, seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        check_sequence(seq)
        b, t, n, c = seq.shape
        x = seq.view(b * t, n, c)
        pt = self.point(x).mean(dim=1).view(b, t, -1)
        temporal, h = self.temporal(pt)
        summary = h[-1]
        return temporal, summary


class QueryTrackHead(nn.Module):
    def __init__(self, *, width: int, num_tracks: int, num_classes: int) -> None:
        super().__init__()
        self.num_tracks = int(num_tracks)
        self.queries = nn.Parameter(torch.randn(int(num_tracks), int(width)) * 0.02)
        self.fuse = nn.Sequential(
            nn.Linear(int(width) * 2, int(width)),
            nn.GELU(),
            nn.Linear(int(width), int(width)),
        )
        self.box_head = nn.Linear(int(width), 7)
        self.score_head = nn.Linear(int(width), 1)
        self.cls_head = nn.Linear(int(width), int(num_classes))

    def forward(self, summary: torch.Tensor, temporal: torch.Tensor) -> dict[str, torch.Tensor]:
        b = summary.shape[0]
        q = self.queries.unsqueeze(0).expand(b, -1, -1)
        seq_ctx = temporal.mean(dim=1, keepdim=True).expand_as(q)
        state = self.fuse(torch.cat([q + summary.unsqueeze(1), seq_ctx], dim=-1))
        boxes = self.box_head(state)
        scores = torch.sigmoid(self.score_head(state)).squeeze(-1)
        cls_logits = self.cls_head(state)
        track_ids = torch.arange(self.num_tracks, device=summary.device).unsqueeze(0).expand(b, -1)
        return {
            "track_boxes": boxes,
            "track_scores": scores,
            "track_ids": track_ids.to(torch.float32),
            "cls_logits": cls_logits,
        }


class KalmanAssociationTracker3D(nn.Module):
    def __init__(
        self,
        *,
        family: str,
        in_channels: int,
        num_classes: int,
        width: int,
        num_tracks: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.family = str(family)
        self.encoder = TemporalPointEncoder(
            in_channels=int(in_channels), width=int(width), dropout=float(dropout)
        )
        self.head = QueryTrackHead(
            width=int(width), num_tracks=int(num_tracks), num_classes=int(num_classes)
        )
        self.kalman_gain = nn.Linear(int(width), 7)
        self.mode_head = nn.Linear(int(width), 3)

    def track(self, seq: torch.Tensor) -> dict[str, torch.Tensor]:
        temporal, summary = self.encoder(seq)
        out = self.head(summary, temporal)
        state = temporal.mean(dim=1)
        gain = torch.tanh(self.kalman_gain(state)).unsqueeze(1)
        out["track_boxes"] = out["track_boxes"] + 0.1 * gain
        if self.family == "imm_kalman":
            out["mode_weights"] = torch.softmax(self.mode_head(state), dim=-1)
        else:
            q = out["cls_logits"].mean(dim=-1)
            out["association_logits"] = q.unsqueeze(2) - q.unsqueeze(1)
        return out


class BEVTracking3D(nn.Module):
    def __init__(
        self,
        *,
        family: str,
        in_channels: int,
        num_classes: int,
        width: int,
        num_tracks: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.family = str(family)
        self.encoder = TemporalPointEncoder(
            in_channels=int(in_channels), width=int(width), dropout=float(dropout)
        )
        self.head = QueryTrackHead(
            width=int(width), num_tracks=int(num_tracks), num_classes=int(num_classes)
        )
        self.motion_head = nn.Linear(int(width), 7)

    def track(self, seq: torch.Tensor) -> dict[str, torch.Tensor]:
        temporal, summary = self.encoder(seq)
        out = self.head(summary, temporal)
        delta = torch.tanh(self.motion_head(temporal[:, -1] - temporal[:, 0])).unsqueeze(1)
        out["track_boxes"] = out["track_boxes"] + 0.08 * delta
        out["motion_logits"] = delta.squeeze(1)
        out["bev_heatmap"] = torch.sigmoid(out["cls_logits"].mean(dim=1))
        return out


class SegTracking3D(nn.Module):
    def __init__(
        self,
        *,
        family: str,
        in_channels: int,
        num_classes: int,
        width: int,
        num_tracks: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.family = str(family)
        self.encoder = TemporalPointEncoder(
            in_channels=int(in_channels), width=int(width), dropout=float(dropout)
        )
        self.head = QueryTrackHead(
            width=int(width), num_tracks=int(num_tracks), num_classes=int(num_classes)
        )
        self.mask_head = nn.Linear(int(width), int(num_tracks))

    def track(self, seq: torch.Tensor) -> dict[str, torch.Tensor]:
        temporal, summary = self.encoder(seq)
        out = self.head(summary, temporal)
        mask_logits = self.mask_head(temporal.mean(dim=1))
        out["mask_logits"] = mask_logits
        out["track_scores"] = out["track_scores"] + 0.05 * torch.sigmoid(
            mask_logits[:, : out["track_scores"].shape[1]]
        )
        return out


def smoke_test_tracker(builder, variant: str) -> None:
    tracker = builder(
        in_channels=3, num_classes=3, seq_len=4, variant=variant, width_mult=0.5, dropout=0.0
    )
    x = torch.randn(2, 4, 128, 3)
    out = tracker.track(x)
    print(variant, {k: tuple(v.shape) for k, v in out.items() if torch.is_tensor(v)})
    assert "track_boxes" in out and "track_scores" in out and "track_ids" in out
    print("ok")
