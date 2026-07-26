from __future__ import annotations
import torch
import torch.nn.functional as F
from torch import nn


def check_nchw(image: torch.Tensor) -> torch.Tensor:
    image = image.to(torch.float32)
    if image.ndim != 4:
        raise ValueError(f"Expected input shape (B, C, H, W), got {tuple(image.shape)}")
    return image


class TinyShadowBlock(nn.Module):
    def __init__(self, *, channels: int, mode: str) -> None:
        super().__init__()
        self.mode = str(mode)
        self.conv1 = nn.Conv2d(int(channels), int(channels), 3, padding=1)
        self.conv2 = nn.Conv2d(int(channels), int(channels), 3, padding=1)
        self.mix = nn.Conv2d(int(channels), int(channels), 1)
        self.norm = nn.GroupNorm(1, int(channels))
        self.depthwise = nn.Conv2d(int(channels), int(channels), 5, padding=2, groups=int(channels))
        self.prompt = (
            nn.Parameter(torch.zeros(1, int(channels), 1, 1)) if self.mode == "prompt" else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        if self.prompt is not None:
            h = h + self.prompt
        local = self.conv2(F.relu(self.conv1(h), inplace=True))
        if self.mode == "bdrar":
            local = local + self.mix(h)
        elif self.mode == "stacked":
            local = local + 0.5 * self.depthwise(h)
        elif self.mode == "context":
            local = local + F.avg_pool2d(h, 3, 1, 1)
        elif self.mode == "boundary":
            local = local + torch.tanh(self.depthwise(h))
        elif self.mode == "transformer":
            local = local * torch.sigmoid(self.mix(h))
        elif self.mode == "diffusion":
            local = 0.7 * local + 0.3 * torch.tanh(self.mix(h))
        elif self.mode == "state_space":
            local = local + torch.roll(h, shifts=1, dims=-1)
        elif self.mode == "mamba":
            local = local + torch.roll(h, shifts=1, dims=-2)
        return x + 0.2 * local


class TinyShadowDetector(nn.Module):
    def __init__(self, *, family: str, mode: str, in_channels: int, width: int, depth: int) -> None:
        super().__init__()
        self.family = str(family)
        self.mode = str(mode)
        self.stem = nn.Conv2d(int(in_channels), int(width), 3, padding=1)
        self.blocks = nn.ModuleList(
            [
                TinyShadowBlock(channels=int(width), mode=str(mode))
                for _ in range(max(1, int(depth)))
            ]
        )
        self.mask_head = nn.Conv2d(int(width), 1, 1)
        self.boundary_head = nn.Conv2d(int(width), 1, 1)

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = F.relu(self.stem(check_nchw(image)), inplace=True)
        for block in self.blocks:
            feat = block(feat)
        logits = self.mask_head(feat)
        return {
            "logits": logits,
            "mask": torch.sigmoid(logits),
            "boundary": torch.sigmoid(self.boundary_head(feat)),
        }


def build_toy_shadow_detector(
    *,
    family: str,
    mode: str,
    variants: dict[str, dict[str, int]],
    in_channels: int,
    variant: str,
    width_mult: float = 1.0,
) -> nn.Module:
    spec = dict(variants[str(variant).lower().strip()])
    width = max(12, int(int(spec["width"]) * float(width_mult)))
    return TinyShadowDetector(
        family=str(family),
        mode=str(mode),
        in_channels=int(in_channels),
        width=width,
        depth=int(spec["depth"]),
    )


def smoke_test_shadow_detector(builder, variant: str) -> None:
    model = builder(in_channels=3, variant=variant, width_mult=0.5)
    out = model(torch.randn(2, 3, 32, 32))
    print(variant, tuple(out["mask"].shape))
