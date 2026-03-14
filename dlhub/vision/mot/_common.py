from __future__ import annotations

import torch
from torch import nn


def check_video(video: torch.Tensor) -> None:
    if not isinstance(video, torch.Tensor):
        raise TypeError(f"video must be a torch.Tensor, got {type(video).__name__}")
    if video.ndim != 5:
        raise ValueError(f"video must have shape (B, T, C, H, W), got {tuple(video.shape)}")
    if video.shape[2] < 1:
        raise ValueError(f"video channel dim must be >= 1, got C={video.shape[2]}")


def _check_detections(detections: torch.Tensor | None, *, batch_size: int, seq_len: int) -> None:
    if detections is None:
        return
    if not isinstance(detections, torch.Tensor):
        raise TypeError(f"detections must be torch.Tensor, got {type(detections).__name__}")
    if detections.ndim != 4 or detections.shape[-1] != 4:
        raise ValueError(
            f"detections must have shape (B, T, N, 4), got {tuple(detections.shape)}"
        )
    if int(detections.shape[0]) != int(batch_size) or int(detections.shape[1]) != int(seq_len):
        raise ValueError(
            "detections batch/temporal dims must match video "
            f"((B,T)=({batch_size},{seq_len})), got {tuple(detections.shape[:2])}"
        )


class FrameEncoder(nn.Module):
    def __init__(self, *, in_channels: int, width: int, dropout: float = 0.0) -> None:
        super().__init__()
        w = int(width)
        self.net = nn.Sequential(
            nn.Conv2d(int(in_channels), w, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(w),
            nn.GELU(),
            nn.Conv2d(w, w, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(w),
            nn.GELU(),
            nn.Conv2d(w, w, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(float(dropout)),
        )

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        check_video(video)
        b, t, c, h, w = video.shape
        x = video.reshape(b * t, c, h, w)
        f = self.net(x).reshape(b, t, -1)
        return f


class MOTTracker2D(nn.Module):
    def __init__(
        self,
        *,
        family: str,
        group: str,
        in_channels: int,
        num_classes: int,
        width: int,
        num_tracks: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.family = str(family)
        self.group = str(group)
        hidden = int(width)
        self.num_tracks = int(num_tracks)
        self.encoder = FrameEncoder(
            in_channels=int(in_channels), width=hidden, dropout=float(dropout)
        )
        self.temporal = nn.GRU(hidden, hidden, num_layers=1, batch_first=True)
        self.queries = nn.Parameter(torch.randn(int(num_tracks), hidden) * 0.02)
        self.fuse = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden, hidden),
        )
        self.box_head = nn.Linear(hidden, 4)
        self.score_head = nn.Linear(hidden, 1)
        self.cls_head = nn.Linear(hidden, int(num_classes))

        # Group-specific tiny heads.
        self.embed_head = nn.Linear(hidden, hidden)
        self.path_head = nn.Linear(hidden, int(num_tracks))
        self.cov_head = nn.Linear(hidden, 4)

    def _build_state(self, encoded: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        temporal, hidden = self.temporal(encoded)
        summary = hidden[-1]
        b = summary.shape[0]
        q = self.queries.unsqueeze(0).expand(b, -1, -1)
        temporal_ctx = temporal.mean(dim=1, keepdim=True).expand_as(q)
        state = self.fuse(torch.cat([q + summary.unsqueeze(1), temporal_ctx], dim=-1))
        return temporal, state

    def track(
        self,
        video: torch.Tensor,
        detections: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        check_video(video)
        b, t, _, _, _ = video.shape
        _check_detections(detections, batch_size=int(b), seq_len=int(t))

        encoded = self.encoder(video)
        temporal, state = self._build_state(encoded)
        boxes = torch.sigmoid(self.box_head(state))
        scores = torch.sigmoid(self.score_head(state)).squeeze(-1)
        cls_logits = self.cls_head(state)
        track_ids = torch.arange(self.num_tracks, device=video.device).unsqueeze(0).expand(b, -1)

        out: dict[str, torch.Tensor] = {
            "track_boxes": boxes,
            "track_scores": scores,
            "track_ids": track_ids.to(torch.float32),
            "cls_logits": cls_logits,
        }

        if self.group == "online_association":
            # Pairwise affinity logits for online assignment.
            pair = scores.unsqueeze(2) - scores.unsqueeze(1)
            out["association_logits"] = pair
        elif self.group == "joint_det_embed":
            emb = nn.functional.normalize(self.embed_head(state), dim=-1)
            out["track_embeddings"] = emb
            out["match_logits"] = torch.matmul(emb, emb.transpose(1, 2))
        elif self.group == "query_transformer":
            out["query_tokens"] = state
            out["temporal_memory"] = temporal
        elif self.group == "global_optimization":
            path = self.path_head(state.mean(dim=1))
            out["path_costs"] = path
        elif self.group == "probabilistic_filtering":
            cov = torch.softplus(self.cov_head(state)) + 1e-4
            out["state_covariance"] = cov

        if detections is not None:
            # Keep a tiny hook to expose detector-track alignment in future iterations.
            det_center = detections[..., :2].mean(dim=2)
            out["det_center_mean"] = det_center
        return out


def smoke_test_tracker(builder, variant: str) -> None:
    tracker = builder(
        in_channels=3,
        num_classes=3,
        seq_len=4,
        image_size=64,
        variant=variant,
        width_mult=0.5,
        dropout=0.0,
    )
    x = torch.randn(2, 4, 3, 64, 64)
    out = tracker.track(x)
    print(variant, {k: tuple(v.shape) for k, v in out.items() if torch.is_tensor(v)})
    assert "track_boxes" in out and "track_scores" in out and "track_ids" in out
    print("ok")
