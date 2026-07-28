"""Shared query-conditioned GRU baseline for temporal-grounding aliases."""

from __future__ import annotations

import torch
from torch import nn


class TemporalGroundingBaseline(nn.Module):
    """Fuse frame/query features and predict normalized temporal boundaries."""

    def __init__(self, *, registered_alias: str, in_channels: int, width: int, depth: int):
        super().__init__()
        self.registered_alias = str(registered_alias)
        self.proj = nn.Linear(int(in_channels), int(width))
        self.temporal = nn.GRU(
            int(width),
            int(width),
            num_layers=max(1, int(depth)),
            batch_first=True,
        )
        self.boundary = nn.Linear(int(width), 2)

    def forward(self, video_feat, query_feat=None):
        x = video_feat.to(torch.float32)
        if x.ndim != 3:
            raise ValueError(f"Expected input shape (B,T,C), got {tuple(x.shape)}")
        q = (
            torch.zeros(x.shape[0], x.shape[2], device=x.device)
            if query_feat is None
            else query_feat.to(x.dtype)
        )
        seq, _ = self.temporal(self.proj(x) + self.proj(q.unsqueeze(1).expand_as(x)))
        return {"boundaries": torch.sigmoid(self.boundary(seq))}


def build_temporal_grounding_baseline(
    *,
    registered_alias: str,
    variants: dict[str, dict[str, int]],
    in_channels: int,
    variant: str,
    width_mult: float = 1.0,
    **kwargs,
):
    """Build the shared baseline behind a registered compatibility alias."""

    spec = variants[str(variant)]
    width = max(16, int(int(spec["width"]) * float(width_mult)))
    return TemporalGroundingBaseline(
        registered_alias=str(registered_alias),
        in_channels=int(in_channels),
        width=width,
        depth=int(spec["depth"]),
    )


def smoke_test_model(builder, variant: str):
    out = builder(in_channels=64, variant=variant, width_mult=0.5)(torch.randn(2, 16, 64))
    print(variant, tuple(out["boundaries"].shape))
