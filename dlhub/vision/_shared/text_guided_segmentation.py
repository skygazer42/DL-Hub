"""Shared toy text-guided segmentation model (open-vocabulary / referring)."""

from __future__ import annotations

import torch
from torch import nn


def check_nchw(x: torch.Tensor) -> torch.Tensor:
    x = x.to(torch.float32)
    if x.ndim != 4:
        raise ValueError(f"Expected input shape (B,C,H,W), got {tuple(x.shape)}")
    return x


class TinyEncoder(nn.Module):
    def __init__(self, in_channels: int, width: int, depth: int) -> None:
        super().__init__()
        c = int(width)
        layers: list[nn.Module] = [nn.Conv2d(int(in_channels), c, 3, 1, 1), nn.ReLU(inplace=True)]
        for _ in range(max(1, int(depth))):
            layers += [nn.Conv2d(c, c, 3, 1, 1), nn.ReLU(inplace=True)]
        self.net = nn.Sequential(*layers)
        self.out_channels = c

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(check_nchw(x))


class ToyModel(nn.Module):
    def __init__(
        self, *, family: str, in_channels: int, width: int, depth: int, text_dim: int = 32
    ) -> None:
        super().__init__()
        self.family = str(family)
        self.enc = TinyEncoder(in_channels, width, depth)
        c = self.enc.out_channels
        self.txt = nn.Linear(text_dim, c)
        self.mask = nn.Conv2d(c, 1, 1)

    def forward(
        self, image: torch.Tensor, text_feat: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        x = check_nchw(image)
        feat = self.enc(x)
        if text_feat is None:
            # Deterministic "no prompt" default; injecting randn here made
            # eval outputs irreproducible (and hardcoded dim 32 crashed
            # when text_dim differed).
            text_feat = torch.zeros(image.shape[0], self.txt.in_features, device=image.device)
        bias = self.txt(text_feat.to(feat.dtype)).unsqueeze(-1).unsqueeze(-1)
        logits = self.mask(feat + bias)
        return {"logits": logits, "mask": torch.sigmoid(logits)}


def build_toy_model(
    *,
    family: str,
    variants: dict[str, dict[str, int]],
    in_channels: int,
    variant: str,
    width_mult: float = 1.0,
    **kwargs,
) -> ToyModel:
    spec = variants[str(variant)]
    width = max(16, int(int(spec["width"]) * float(width_mult)))
    return ToyModel(
        family=str(family), in_channels=int(in_channels), width=width, depth=int(spec["depth"])
    )


def smoke_test_model(builder, variant: str) -> None:
    out = builder(in_channels=3, variant=variant, width_mult=0.5)(torch.randn(2, 3, 64, 64))
    print(variant, tuple(out["mask"].shape))
