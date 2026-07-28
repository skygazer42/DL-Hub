"""Shared GRU baseline for temporal action-localization aliases."""

from __future__ import annotations

import torch
from torch import nn


class TemporalActionLocalizationBaseline(nn.Module):
    """Project frame features through a stacked GRU and prediction heads."""

    def __init__(
        self,
        *,
        registered_alias: str,
        in_channels: int,
        width: int,
        depth: int,
        num_classes: int = 5,
    ):
        super().__init__()
        self.registered_alias = str(registered_alias)
        self.proj = nn.Linear(int(in_channels), int(width))
        self.temporal = nn.GRU(
            int(width),
            int(width),
            num_layers=max(1, int(depth)),
            batch_first=True,
        )
        self.cls = nn.Linear(int(width), int(num_classes))
        self.boundary = nn.Linear(int(width), 2)

    def forward(self, video_feat):
        x = video_feat.to(torch.float32)
        if x.ndim != 3:
            raise ValueError(f"Expected input shape (B,T,C), got {tuple(x.shape)}")
        tok = self.proj(x)
        seq, _ = self.temporal(tok)
        return {"class_logits": self.cls(seq), "boundaries": torch.sigmoid(self.boundary(seq))}


def build_temporal_action_localization_baseline(
    *,
    registered_alias: str,
    variants: dict[str, dict[str, int]],
    in_channels: int,
    variant: str,
    width_mult: float = 1.0,
    num_classes: int = 5,
    **kwargs,
):
    """Build the shared baseline behind a registered compatibility alias."""

    spec = variants[str(variant)]
    width = max(16, int(int(spec["width"]) * float(width_mult)))
    return TemporalActionLocalizationBaseline(
        registered_alias=str(registered_alias),
        in_channels=int(in_channels),
        width=width,
        depth=int(spec["depth"]),
        num_classes=int(num_classes),
    )


def smoke_test_model(builder, variant: str):
    out = builder(in_channels=64, variant=variant, width_mult=0.5)(torch.randn(2, 16, 64))
    print(variant, {k: tuple(v.shape) for k, v in out.items()})
