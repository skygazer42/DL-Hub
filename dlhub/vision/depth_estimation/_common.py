from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def check_nchw(x: torch.Tensor) -> torch.Tensor:
    x = x.to(torch.float32)
    if x.ndim != 4:
        raise ValueError(f"Expected input shape (B, C, H, W), got {tuple(x.shape)}")
    return x


class TinyDepthEstimator(nn.Module):
    def __init__(
        self, *, family: str, in_channels: int, width: int, depth: int, bins: bool = False
    ) -> None:
        super().__init__()
        self.family = str(family)
        self.bins = bool(bins)
        c = int(width)
        layers: list[nn.Module] = [nn.Conv2d(int(in_channels), c, 3, 1, 1), nn.ReLU(inplace=True)]
        for _ in range(max(1, int(depth))):
            layers += [nn.Conv2d(c, c, 3, 1, 1), nn.ReLU(inplace=True)]
        self.net = nn.Sequential(*layers)
        self.head = nn.Conv2d(c, 32 if self.bins else 1, 1)

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.net(check_nchw(image))
        raw = self.head(x)
        if self.bins:
            prob = torch.softmax(raw, dim=1)
            centers = torch.linspace(
                0.1, 10.0, steps=prob.shape[1], device=prob.device, dtype=prob.dtype
            ).view(1, -1, 1, 1)
            depth = (prob * centers).sum(dim=1, keepdim=True)
            return {"depth": depth, "bin_logits": raw}
        depth = F.softplus(raw) + 1e-3
        return {"depth": depth}


def build_baseline_depth_estimator(
    *,
    family: str,
    variants: dict[str, dict[str, int]],
    in_channels: int,
    variant: str,
    width_mult: float = 1.0,
    bins: bool = False,
) -> nn.Module:
    spec = variants[str(variant)]
    width = max(16, int(int(spec["width"]) * float(width_mult)))
    return TinyDepthEstimator(
        family=str(family),
        in_channels=int(in_channels),
        width=width,
        depth=int(spec["depth"]),
        bins=bool(bins),
    )


def smoke_test_depth(builder, variant: str) -> None:
    model = builder(in_channels=3, variant=variant, width_mult=0.5)
    x = torch.randn(2, 3, 64, 64)
    out = model(x)
    print(variant, {k: tuple(v.shape) for k, v in out.items()})
    print("ok")
