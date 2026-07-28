from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def check_nchw(image: torch.Tensor) -> torch.Tensor:
    image = image.to(torch.float32)
    if image.ndim != 4:
        raise ValueError(f"Expected input shape (B, C, H, W), got {tuple(image.shape)}")
    return image


class TinyUWIEBlock(nn.Module):
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

    def forward(self, x: torch.Tensor, aux: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        if self.prompt is not None:
            h = h + self.prompt
        local = self.conv2(F.relu(self.conv1(h), inplace=True))
        if self.mode == "color_cast":
            local = local + 0.3 * aux
        elif self.mode == "haze_compensation":
            local = local + self.mix(aux)
        elif self.mode == "white_balance":
            local = local + (h - h.mean(dim=(2, 3), keepdim=True))
        elif self.mode == "contrast":
            local = local + 0.5 * (h - F.avg_pool2d(h, kernel_size=3, stride=1, padding=1))
        elif self.mode == "retinex":
            low = F.avg_pool2d(h, kernel_size=7, stride=1, padding=3)
            local = local + (h - low)
        elif self.mode == "fusion":
            local = local + self.depthwise(h) + 0.2 * aux
        elif self.mode == "transformer":
            local = local * torch.sigmoid(self.mix(h)) + self.depthwise(h)
        elif self.mode == "frequency":
            smooth = F.avg_pool2d(h, kernel_size=5, stride=1, padding=2)
            local = local + (h - smooth)
        elif self.mode == "prompt":
            local = local + self.mix(aux)
        elif self.mode == "mamba":
            local = local + torch.tanh(self.depthwise(torch.roll(h, shifts=1, dims=-1)))
        return x + 0.2 * local


class TinyUWIEEnhancer(nn.Module):
    def __init__(
        self,
        *,
        family: str,
        mode: str,
        in_channels: int,
        width: int,
        depth: int,
        steps: int,
    ) -> None:
        super().__init__()
        self.family = str(family)
        self.mode = str(mode)
        self.steps = max(1, int(steps))
        self.encoder = nn.Conv2d(int(in_channels), int(width), kernel_size=3, padding=1)
        self.aux_encoder = nn.Conv2d(int(in_channels), int(width), kernel_size=1)
        self.blocks = nn.ModuleList(
            [TinyUWIEBlock(channels=int(width), mode=str(mode)) for _ in range(max(1, int(depth)))]
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(int(width), int(width), kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(width), int(in_channels), kernel_size=3, padding=1),
        )
        self.depth_head = nn.Conv2d(int(width), 1, kernel_size=1)
        self.trans_head = nn.Conv2d(int(width), 1, kernel_size=1)

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        image = check_nchw(image)
        enhanced = image
        for _ in range(self.steps):
            feat = self.encoder(enhanced)
            aux = self.aux_encoder(image - enhanced)
            for block in self.blocks:
                feat = block(feat, aux)
            residual = self.decoder(feat)
            if self.mode == "white_balance":
                residual = residual - residual.mean(dim=1, keepdim=True)
            elif self.mode == "contrast":
                residual = 0.6 * residual + 0.4 * (enhanced - F.avg_pool2d(enhanced, 3, 1, 1))
            elif self.mode == "frequency":
                residual = 0.5 * residual + 0.5 * torch.tanh(residual)
            enhanced = torch.clamp(enhanced - residual, -1.0, 1.0)

        feat_final = self.encoder(enhanced)
        attenuation_map = torch.clamp(image - enhanced, -1.0, 1.0)
        depth_map = torch.sigmoid(self.depth_head(feat_final))
        transmission_map = torch.sigmoid(self.trans_head(feat_final))
        return {
            "enhanced": enhanced,
            "attenuation_map": attenuation_map,
            "depth_map": depth_map,
            "transmission_map": transmission_map,
        }


def build_baseline_enhancer(
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
    steps = int(spec.get("steps", 1))
    return TinyUWIEEnhancer(
        family=str(family),
        mode=str(mode),
        in_channels=int(in_channels),
        width=width,
        depth=depth,
        steps=steps,
    )


def smoke_test_enhancer(builder, variant: str) -> None:
    model = builder(in_channels=3, variant=variant, width_mult=0.5)
    image = torch.randn(2, 3, 32, 32)
    out = model(image)
    print(variant, tuple(out["enhanced"].shape), tuple(out["depth_map"].shape))
    print("ok")
