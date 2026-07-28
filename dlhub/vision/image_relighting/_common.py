from __future__ import annotations

import torch
from torch import nn


def check_nchw(x: torch.Tensor) -> torch.Tensor:
    x = x.to(torch.float32)
    if x.ndim != 4:
        raise ValueError(f"Expected input shape (B,C,H,W), got {tuple(x.shape)}")
    return x


class TinyEncoder(nn.Module):
    def __init__(self, in_channels: int, width: int, depth: int):
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


class CompactRelighter(nn.Module):
    def __init__(self, *, family: str, in_channels: int, width: int, depth: int):
        super().__init__()
        self.family = str(family)
        self.encoder = TinyEncoder(in_channels=in_channels, width=width, depth=depth)
        channels = self.encoder.out_channels
        self.light_head = nn.Conv2d(channels, 1, kernel_size=1)
        self.gain_head = nn.Conv2d(channels, int(in_channels), kernel_size=1)
        self.residual_head = nn.Conv2d(channels, int(in_channels), kernel_size=1)

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        x = check_nchw(image)
        feat = self.encoder(x)
        light_map = torch.sigmoid(self.light_head(feat))
        channel_gain = 0.75 + 0.5 * torch.sigmoid(self.gain_head(feat))
        residual = torch.tanh(self.residual_head(feat))
        relit = torch.clamp(x * channel_gain * (0.8 + 0.4 * light_map) + 0.15 * residual, 0.0, 1.0)
        return {"relit": relit, "light_map": light_map, "residual": residual}


def build_baseline_model(
    *,
    family: str,
    variants: dict[str, dict[str, int]],
    in_channels: int,
    variant: str,
    width_mult: float = 1.0,
    **kwargs,
) -> CompactRelighter:
    del kwargs
    name = str(variant)
    if name not in variants:
        raise ValueError(f"Unknown {family} variant: {variant!r}. Supported: {sorted(variants)}")
    spec = variants[name]
    width = max(16, int(int(spec["width"]) * float(width_mult)))
    return CompactRelighter(
        family=str(family),
        in_channels=int(in_channels),
        width=width,
        depth=int(spec["depth"]),
    )


def smoke_test_model(builder, variant: str) -> None:
    model = builder(in_channels=3, variant=variant, width_mult=0.5)
    out = model(torch.rand(2, 3, 64, 64))
    print(variant, {k: tuple(v.shape) for k, v in out.items()})
