from __future__ import annotations

import torch
from torch import nn


class CompactVideoMatter(nn.Module):
    def __init__(self, *, in_channels: int, width: int, depth: int):
        super().__init__()
        c = int(in_channels)
        hidden = int(width)
        layers: list[nn.Module] = []
        for _ in range(int(depth)):
            layers.append(nn.Conv2d(c, hidden, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))
            c = hidden
        self.encoder = nn.Sequential(*layers)
        self.alpha_head = nn.Conv2d(c, 1, kernel_size=1)
        self.foreground_head = nn.Conv2d(c, int(in_channels), kernel_size=1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = x.to(torch.float32)
        if x.ndim != 4:
            raise ValueError(f"Expected input shape (B,C,H,W), got {tuple(x.shape)}")
        feat = self.encoder(x)
        alpha = torch.sigmoid(self.alpha_head(feat))
        foreground = torch.tanh(self.foreground_head(feat))
        composite = alpha * foreground
        return {"alpha": alpha, "foreground": foreground, "composite": composite}


def build_baseline_video_matter(
    *,
    variants: dict[str, dict[str, int]],
    in_channels: int,
    variant: str,
    width_mult: float = 1.0,
):
    spec = dict(variants[str(variant)])
    width = max(8, int(spec["width"] * float(width_mult)))
    depth = int(spec["depth"])
    return CompactVideoMatter(in_channels=int(in_channels), width=width, depth=depth)


def smoke_test_video_matter(builder, variant: str) -> None:
    model = builder(in_channels=3, variant=variant)
    out = model(torch.randn(2, 3, 32, 32))
    print(variant, {k: tuple(v.shape) for k, v in out.items()})
