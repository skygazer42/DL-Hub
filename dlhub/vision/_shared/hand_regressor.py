from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def check_nchw(image: torch.Tensor) -> torch.Tensor:
    image = image.to(torch.float32)
    if image.ndim != 4:
        raise ValueError(f"Expected input shape (B, C, H, W), got {tuple(image.shape)}")
    return image


class TinyHandRegressorBlock(nn.Module):
    # Sub-layers are only created for the modes whose forward branch uses
    # them; unconditional creation left dead (never-trained) parameters on
    # every mode except "transformer".
    _USES_MIX = {"transformer", "coarse_to_fine", "uncertainty"}
    _USES_DEPTHWISE = {"direct", "heatmap", "graph", "skeleton", "region", "transformer", "mamba"}

    def __init__(self, *, channels: int, mode: str) -> None:
        super().__init__()
        self.mode = str(mode)
        self.norm = nn.GroupNorm(1, int(channels))
        self.conv1 = nn.Conv2d(int(channels), int(channels), kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(int(channels), int(channels), kernel_size=3, padding=1)
        self.mix = (
            nn.Conv2d(int(channels), int(channels), kernel_size=1)
            if self.mode in self._USES_MIX
            else None
        )
        self.depthwise = (
            nn.Conv2d(int(channels), int(channels), kernel_size=5, padding=2, groups=int(channels))
            if self.mode in self._USES_DEPTHWISE
            else None
        )
        self.prompt = (
            nn.Parameter(torch.zeros(1, int(channels), 1, 1)) if self.mode == "prompt" else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        if self.prompt is not None:
            h = h + self.prompt
        local = self.conv2(F.relu(self.conv1(h), inplace=True))
        if self.mode in {"direct", "heatmap", "graph", "skeleton", "region"}:
            local = local + self.depthwise(h)
        elif self.mode == "transformer":
            local = local * torch.sigmoid(self.mix(h)) + self.depthwise(h)
        elif self.mode == "coarse_to_fine":
            local = local + self.mix(F.avg_pool2d(h, 3, 1, 1))
        elif self.mode == "multiview":
            local = local + torch.roll(h, shifts=1, dims=-1)
        elif self.mode == "uncertainty":
            local = local + torch.tanh(self.mix(h))
        elif self.mode == "mamba":
            local = local + torch.tanh(self.depthwise(torch.roll(h, shifts=1, dims=-2)))
        return x + 0.2 * local


class TinyHandRegressor(nn.Module):
    def __init__(self, *, family: str, mode: str, in_channels: int, width: int, depth: int) -> None:
        super().__init__()
        self.family = str(family)
        self.mode = str(mode)
        self.stem = nn.Conv2d(int(in_channels), int(width), kernel_size=3, padding=1)
        self.blocks = nn.ModuleList(
            [
                TinyHandRegressorBlock(channels=int(width), mode=str(mode))
                for _ in range(max(1, int(depth)))
            ]
        )
        self.value_head = nn.Linear(int(width), 1)
        self.conf_head = nn.Linear(int(width), 1)
        self.map_head = nn.Conv2d(int(width), 1, kernel_size=1)

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        image = check_nchw(image)
        feat = F.relu(self.stem(image), inplace=True)
        for block in self.blocks:
            feat = block(feat)
        pooled = F.adaptive_avg_pool2d(feat, 1).flatten(1)
        return {
            "value": torch.tanh(self.value_head(pooled)),
            "confidence": torch.sigmoid(self.conf_head(pooled)),
            "map": self.map_head(feat),
        }


def build_baseline_hand_regressor(
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
    return TinyHandRegressor(
        family=str(family), mode=str(mode), in_channels=int(in_channels), width=width, depth=depth
    )


def smoke_test_hand_regressor(builder, variant: str) -> None:
    model = builder(in_channels=3, variant=variant, width_mult=0.5)
    out = model(torch.randn(2, 3, 64, 64))
    print(variant, tuple(out["value"].shape), tuple(out["map"].shape))
    print("ok")
