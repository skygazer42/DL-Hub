from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def check_nchw(image: torch.Tensor) -> torch.Tensor:
    image = image.to(torch.float32)
    if image.ndim != 4:
        raise ValueError(f"Expected input shape (B, C, H, W), got {tuple(image.shape)}")
    return image


class TinyMotionBlock(nn.Module):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        if self.prompt is not None:
            h = h + self.prompt
        local = self.conv2(F.relu(self.conv1(h), inplace=True))
        if self.mode in {"flow", "difference", "recurrent", "contour"}:
            local = local + self.depthwise(h)
        elif self.mode == "transformer":
            local = local * torch.sigmoid(self.mix(h)) + self.depthwise(h)
        elif self.mode == "dual":
            local = local + self.mix(torch.roll(h, shifts=1, dims=-1))
        elif self.mode == "pyramid":
            local = local + self.mix(F.avg_pool2d(h, 3, 1, 1))
        elif self.mode == "coarse":
            local = local + self.mix(torch.roll(h, shifts=1, dims=-2))
        elif self.mode == "mamba":
            local = local + torch.tanh(self.depthwise(torch.roll(h, shifts=1, dims=-1)))
        return x + 0.2 * local


class TinyMotionSegmentor(nn.Module):
    def __init__(self, *, family: str, mode: str, in_channels: int, width: int, depth: int) -> None:
        super().__init__()
        self.family = str(family)
        self.mode = str(mode)
        self.stem = nn.Conv2d(int(in_channels) * 2, int(width), 3, padding=1)
        self.blocks = nn.ModuleList(
            [
                TinyMotionBlock(channels=int(width), mode=str(mode))
                for _ in range(max(1, int(depth)))
            ]
        )
        self.mask_head = nn.Conv2d(int(width), 2, 1)
        self.flow_head = nn.Conv2d(int(width), 2, 1)
        self.boundary_head = nn.Conv2d(int(width), 1, 1)

    def forward(
        self, image: torch.Tensor, reference: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        image = check_nchw(image)
        if reference is None:
            reference = torch.roll(image, shifts=1, dims=-1)
        reference = check_nchw(reference)
        pair = torch.cat([image, reference], dim=1)
        feat = F.relu(self.stem(pair), inplace=True)
        for block in self.blocks:
            feat = block(feat)
        logits = self.mask_head(feat)
        return {
            "logits": logits,
            "mask": torch.softmax(logits, dim=1),
            "flow": self.flow_head(feat),
            "boundary": self.boundary_head(feat),
        }


def build_baseline_motion_segmentor(
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
    width = max(16, int(int(spec["width"]) * float(width_mult)))
    depth = int(spec["depth"])
    return TinyMotionSegmentor(
        family=str(family), mode=str(mode), in_channels=int(in_channels), width=width, depth=depth
    )


def smoke_test_motion_segmentor(builder, variant: str) -> None:
    model = builder(in_channels=3, variant=variant, width_mult=0.5)
    image = torch.randn(2, 3, 64, 64)
    out = model(image, torch.flip(image, dims=(-1,)))
    print(variant, tuple(out["logits"].shape), tuple(out["flow"].shape))
    print("ok")
