from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def check_nchw(image: torch.Tensor) -> torch.Tensor:
    image = image.to(torch.float32)
    if image.ndim != 4:
        raise ValueError(f"Expected input shape (B, C, H, W), got {tuple(image.shape)}")
    return image


class TinyDeweatherBlock(nn.Module):
    def __init__(self, *, channels: int, mode: str) -> None:
        super().__init__()
        self.mode = str(mode)
        self.norm = nn.GroupNorm(1, int(channels))
        self.conv1 = nn.Conv2d(int(channels), int(channels), kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(int(channels), int(channels), kernel_size=3, padding=1)
        self.mix = nn.Conv2d(int(channels), int(channels), kernel_size=1)
        self.depthwise = nn.Conv2d(
            int(channels),
            int(channels),
            kernel_size=5,
            padding=2,
            groups=int(channels),
        )
        self.prompt = (
            nn.Parameter(torch.zeros(1, int(channels), 1, 1)) if self.mode == "prompt" else None
        )

    def forward(self, x: torch.Tensor, weather: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        if self.prompt is not None:
            h = h + self.prompt
        local = self.conv2(F.relu(self.conv1(h), inplace=True))
        if self.mode == "snow":
            local = local + 0.35 * weather
        elif self.mode == "raindrop":
            local = local + self.mix(weather)
        elif self.mode == "fog_streak":
            smooth = F.avg_pool2d(weather, kernel_size=3, stride=1, padding=1)
            local = local + smooth
        elif self.mode == "all_weather":
            local = local + 0.5 * self.mix(h) + 0.25 * weather
        elif self.mode == "cnn":
            local = local + self.depthwise(h)
        elif self.mode == "transformer":
            local = local * torch.sigmoid(self.mix(h)) + self.depthwise(h)
        elif self.mode == "frequency":
            low = F.avg_pool2d(h, kernel_size=5, stride=1, padding=2)
            local = local + (h - low)
        elif self.mode == "diffusion":
            local = 0.7 * local + 0.3 * torch.tanh(self.mix(weather))
        elif self.mode == "prompt":
            local = local + self.mix(weather)
        elif self.mode == "mamba":
            local = local + torch.tanh(self.depthwise(torch.roll(h, shifts=1, dims=-1)))
        return x + 0.2 * local


class TinyDeweatherer(nn.Module):
    def __init__(
        self,
        *,
        family: str,
        mode: str,
        in_channels: int,
        width: int,
        depth: int,
        passes: int,
    ) -> None:
        super().__init__()
        self.family = str(family)
        self.mode = str(mode)
        self.passes = max(1, int(passes))
        self.encoder = nn.Conv2d(int(in_channels), int(width), kernel_size=3, padding=1)
        self.weather_encoder = nn.Conv2d(int(in_channels), int(width), kernel_size=1)
        self.blocks = nn.ModuleList(
            [
                TinyDeweatherBlock(channels=int(width), mode=str(mode))
                for _ in range(max(1, int(depth)))
            ]
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(int(width), int(width), kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(width), int(in_channels), kernel_size=3, padding=1),
        )

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        image = check_nchw(image)
        restored = image
        for _ in range(self.passes):
            feat = self.encoder(restored)
            weather = self.weather_encoder(image - restored)
            for block in self.blocks:
                feat = block(feat, weather)
            residual = self.decoder(feat)
            if self.mode == "snow":
                residual = torch.sigmoid(residual) * residual
            elif self.mode == "fog_streak":
                residual = 0.5 * residual + 0.5 * (restored - F.avg_pool2d(restored, 3, 1, 1))
            elif self.mode == "diffusion":
                residual = 0.5 * residual + 0.5 * torch.tanh(residual)
            restored = torch.clamp(restored - residual, -1.0, 1.0)
        weather_residual = torch.clamp(image - restored, -1.0, 1.0)
        return {"restored": restored, "weather_residual": weather_residual}


def build_toy_deweatherer(
    *,
    family: str,
    mode: str,
    variants: dict[str, dict[str, int]],
    in_channels: int,
    variant: str,
    width_mult: float = 1.0,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in variants:
        raise ValueError(
            f"Unknown variant for {family}: {variant!r}. Available: {sorted(variants)}"
        )
    spec = dict(variants[name])
    width = max(12, int(int(spec["width"]) * float(width_mult)))
    depth = int(spec["depth"])
    passes = int(spec.get("passes", 1))
    return TinyDeweatherer(
        family=str(family),
        mode=str(mode),
        in_channels=int(in_channels),
        width=width,
        depth=depth,
        passes=passes,
    )


def smoke_test_deweatherer(builder, variant: str) -> None:
    model = builder(in_channels=3, variant=variant, width_mult=0.5)
    image = torch.randn(2, 3, 32, 32)
    out = model(image)
    print(variant, tuple(out["restored"].shape))
    print("ok")
