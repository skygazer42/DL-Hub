from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def check_nchw(image: torch.Tensor) -> torch.Tensor:
    image = image.to(torch.float32)
    if image.ndim != 4:
        raise ValueError(f"Expected input shape (B, C, H, W), got {tuple(image.shape)}")
    return image


class TinyFlareBlock(nn.Module):
    def __init__(self, *, channels: int, mode: str) -> None:
        super().__init__()
        self.mode = str(mode)
        self.norm = nn.GroupNorm(1, int(channels))
        self.conv1 = nn.Conv2d(int(channels), int(channels), 3, padding=1)
        self.conv2 = nn.Conv2d(int(channels), int(channels), 3, padding=1)
        self.mix = nn.Conv2d(int(channels), int(channels), 1)
        self.depthwise = nn.Conv2d(int(channels), int(channels), 5, padding=2, groups=int(channels))
        self.prompt = (
            nn.Parameter(torch.zeros(1, int(channels), 1, 1)) if self.mode == "prompt" else None
        )

    def forward(self, x: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        if self.prompt is not None:
            h = h + self.prompt
        local = self.conv2(F.relu(self.conv1(h), inplace=True))
        if self.mode in {"glow", "streak", "halo", "residual", "context", "frequency"}:
            local = local + self.depthwise(h)
        elif self.mode == "transformer":
            local = local * torch.sigmoid(self.mix(h)) + self.depthwise(h)
        elif self.mode == "dual":
            local = local + self.mix(guide)
        elif self.mode == "mamba":
            local = local + torch.tanh(self.depthwise(torch.roll(h, shifts=1, dims=-1)))
        return x + 0.2 * local


class TinyFlareRemover(nn.Module):
    def __init__(
        self, *, family: str, mode: str, in_channels: int, width: int, depth: int, steps: int
    ) -> None:
        super().__init__()
        self.family = str(family)
        self.mode = str(mode)
        self.steps = max(1, int(steps))
        self.encoder = nn.Conv2d(int(in_channels), int(width), 3, padding=1)
        self.guide = nn.Conv2d(int(in_channels), int(width), 1)
        self.blocks = nn.ModuleList(
            [TinyFlareBlock(channels=int(width), mode=str(mode)) for _ in range(max(1, int(depth)))]
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(int(width), int(width), 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(width), int(in_channels), 3, padding=1),
        )
        self.map_head = nn.Conv2d(int(width), 1, 1)

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        image = check_nchw(image)
        restored = image
        for _ in range(self.steps):
            flare_guess = torch.clamp(restored - F.avg_pool2d(restored, 5, 1, 2), -1.0, 1.0)
            feat = F.relu(self.encoder(restored), inplace=True)
            guide = self.guide(flare_guess)
            for block in self.blocks:
                feat = block(feat, guide)
            residual = self.decoder(feat)
            restored = torch.clamp(restored - residual, -1.0, 1.0)
        flare_map = torch.sigmoid(self.map_head(F.relu(self.encoder(restored), inplace=True)))
        return {"restored": restored, "flare_map": flare_map, "residual": restored - image}


def build_toy_flare_remover(
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
    return TinyFlareRemover(
        family=str(family),
        mode=str(mode),
        in_channels=int(in_channels),
        width=width,
        depth=depth,
        steps=steps,
    )


def smoke_test_flare_remover(builder, variant: str) -> None:
    model = builder(in_channels=3, variant=variant, width_mult=0.5)
    out = model(torch.randn(2, 3, 32, 32))
    print(variant, tuple(out["restored"].shape), tuple(out["flare_map"].shape))
    print("ok")
