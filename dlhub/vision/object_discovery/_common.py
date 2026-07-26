from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def check_nchw(image: torch.Tensor) -> torch.Tensor:
    image = image.to(torch.float32)
    if image.ndim != 4:
        raise ValueError(f"Expected input shape (B, C, H, W), got {tuple(image.shape)}")
    return image


class TinyObjectDiscoveryBlock(nn.Module):
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
        if self.mode in {"slot", "cluster", "mask", "region", "part"}:
            local = local + self.depthwise(h)
        elif self.mode == "transformer":
            local = local * torch.sigmoid(self.mix(h)) + self.depthwise(h)
        elif self.mode == "query":
            local = local + self.mix(F.avg_pool2d(h, 3, 1, 1))
        elif self.mode == "dual":
            local = local + self.mix(torch.roll(h, shifts=1, dims=-1))
        elif self.mode == "mamba":
            local = local + torch.tanh(self.depthwise(torch.roll(h, shifts=1, dims=-2)))
        return x + 0.2 * local


class TinyObjectDiscoverer(nn.Module):
    def __init__(
        self,
        *,
        family: str,
        mode: str,
        in_channels: int,
        width: int,
        depth: int,
        num_slots: int = 8,
    ) -> None:
        super().__init__()
        self.family = str(family)
        self.mode = str(mode)
        self.stem = nn.Conv2d(int(in_channels), int(width), 3, padding=1)
        self.blocks = nn.ModuleList(
            [
                TinyObjectDiscoveryBlock(channels=int(width), mode=str(mode))
                for _ in range(max(1, int(depth)))
            ]
        )
        self.slots = nn.Parameter(torch.randn(1, int(num_slots), int(width)) * 0.02)
        self.proj = nn.Linear(int(width), int(width))
        self.mask_head = nn.Conv2d(int(width), int(num_slots), 1)
        self.box_head = nn.Linear(int(width), 4)
        self.score_head = nn.Linear(int(width), 1)

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        image = check_nchw(image)
        feat = F.relu(self.stem(image), inplace=True)
        for block in self.blocks:
            feat = block(feat)
        pooled = F.adaptive_avg_pool2d(feat, 1).flatten(1)
        slots = self.slots.expand(image.shape[0], -1, -1) + self.proj(pooled).unsqueeze(1)
        return {
            "masks": torch.sigmoid(self.mask_head(feat)),
            "slots": slots,
            "pred_boxes": torch.sigmoid(self.box_head(slots)),
            "objectness": torch.sigmoid(self.score_head(slots)).squeeze(-1),
        }


def build_toy_object_discoverer(
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
    num_slots = int(spec.get("slots", 8))
    return TinyObjectDiscoverer(
        family=str(family),
        mode=str(mode),
        in_channels=int(in_channels),
        width=width,
        depth=depth,
        num_slots=num_slots,
    )


def smoke_test_object_discoverer(builder, variant: str) -> None:
    model = builder(in_channels=3, variant=variant, width_mult=0.5)
    out = model(torch.randn(2, 3, 64, 64))
    print(variant, tuple(out["masks"].shape), tuple(out["slots"].shape))
    print("ok")
