"""Shared convolutional baseline for referring-expression aliases."""

from __future__ import annotations

import torch
from torch import nn


def check_nchw(x):
    x = x.to(torch.float32)
    if x.ndim != 4:
        raise ValueError(f"Expected input shape (B,C,H,W), got {tuple(x.shape)}")
    return x


class CompactImageEncoder(nn.Module):
    """Encode an image with a configurable compact convolutional stack."""

    def __init__(self, in_channels: int, width: int, depth: int):
        super().__init__()
        c = int(width)
        layers = [nn.Conv2d(int(in_channels), c, 3, 1, 1), nn.ReLU(inplace=True)]
        for _ in range(max(1, int(depth))):
            layers += [nn.Conv2d(c, c, 3, 1, 1), nn.ReLU(inplace=True)]
        self.net = nn.Sequential(*layers)
        self.out_channels = c

    def forward(self, x):
        return self.net(check_nchw(x))


class ReferringExpressionBaseline(nn.Module):
    """Fuse pooled image features with one text vector and regress a box."""

    def __init__(
        self, *, registered_alias: str, in_channels: int, width: int, depth: int, text_dim: int = 32
    ):
        super().__init__()
        self.registered_alias = str(registered_alias)
        self.enc = CompactImageEncoder(in_channels, width, depth)
        c = self.enc.out_channels
        self.txt = nn.Linear(text_dim, c)
        self.box = nn.Linear(c, 4)

    def forward(self, image, text_feat=None):
        x = check_nchw(image)
        feat = self.enc(x).mean(dim=(2, 3))
        if text_feat is None:
            # Deterministic "no text" default; randn here made eval outputs
            # irreproducible and the hardcoded dim crashed when text_dim != 32.
            text_feat = torch.zeros(image.shape[0], self.txt.in_features, device=image.device)
        fused = feat + self.txt(text_feat.to(feat.dtype))
        return {"boxes": torch.sigmoid(self.box(fused))}


def build_referring_expression_baseline(
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
    return ReferringExpressionBaseline(
        registered_alias=str(registered_alias),
        in_channels=int(in_channels),
        width=width,
        depth=int(spec["depth"]),
    )


def smoke_test_model(builder, variant: str):
    out = builder(in_channels=3, variant=variant, width_mult=0.5)(torch.randn(2, 3, 64, 64))
    print(variant, tuple(out["boxes"].shape))
