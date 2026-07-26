from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def check_nchw(x: torch.Tensor) -> torch.Tensor:
    x = x.to(torch.float32)
    if x.ndim != 4:
        raise ValueError(f"Expected input shape (B,C,H,W), got {tuple(x.shape)}")
    return x


class TinyEncoder(nn.Module):
    def __init__(self, in_channels: int, width: int, depth: int) -> None:
        super().__init__()
        channels = int(width)
        layers: list[nn.Module] = [
            nn.Conv2d(int(in_channels), channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        ]
        for _ in range(max(1, int(depth))):
            layers.extend(
                [
                    nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(inplace=True),
                ]
            )
        self.net = nn.Sequential(*layers)
        self.out_channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(check_nchw(x))


class ToyTransparentSegmenter(nn.Module):
    def __init__(self, *, family: str, in_channels: int, width: int, depth: int) -> None:
        super().__init__()
        self.family = str(family)
        self.encoder = TinyEncoder(in_channels=in_channels, width=width, depth=depth)
        channels = int(self.encoder.out_channels)
        self.mask_head = nn.Conv2d(channels, 1, kernel_size=1)
        self.alpha_head = nn.Conv2d(channels, 1, kernel_size=1)
        self.refraction_head = nn.Conv2d(channels, int(in_channels), kernel_size=1)

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        x = check_nchw(image)
        feat = self.encoder(x)
        logits = self.mask_head(feat)
        alpha = torch.sigmoid(self.alpha_head(feat))
        boundary = torch.sigmoid(
            torch.abs(logits - F.avg_pool2d(logits, kernel_size=3, stride=1, padding=1))
        )
        refraction = torch.tanh(self.refraction_head(feat))
        mask = torch.sigmoid(logits)
        composite = torch.clamp(x + 0.1 * refraction * alpha, 0.0, 1.0)
        return {
            "logits": logits,
            "mask": mask,
            "alpha": alpha,
            "boundary": boundary,
            "composite": composite,
        }


def build_toy_model(
    *,
    family: str,
    variants: dict[str, dict[str, int]],
    in_channels: int,
    variant: str,
    width_mult: float = 1.0,
    **kwargs,
) -> ToyTransparentSegmenter:
    del kwargs
    name = str(variant)
    if name not in variants:
        raise ValueError(f"Unknown {family} variant: {variant!r}. Supported: {sorted(variants)}")
    spec = variants[name]
    width = max(16, int(int(spec["width"]) * float(width_mult)))
    return ToyTransparentSegmenter(
        family=str(family),
        in_channels=int(in_channels),
        width=width,
        depth=int(spec["depth"]),
    )


def smoke_test_model(builder, variant: str) -> None:
    model = builder(in_channels=3, variant=variant, width_mult=0.5)
    out = model(torch.rand(2, 3, 64, 64))
    print(variant, {k: tuple(v.shape) for k, v in out.items()})
