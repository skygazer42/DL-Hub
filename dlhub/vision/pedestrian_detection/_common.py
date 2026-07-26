from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def check_nchw(image: torch.Tensor) -> torch.Tensor:
    image = image.to(torch.float32)
    if image.ndim != 4:
        raise ValueError(f"Expected input shape (B, C, H, W), got {tuple(image.shape)}")
    return image


class TinyPedestrianBlock(nn.Module):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        if self.prompt is not None:
            h = h + self.prompt
        local = self.conv2(F.relu(self.conv1(h), inplace=True))
        if self.mode == "fcos":
            local = local + self.depthwise(h)
        elif self.mode == "center":
            local = local + self.mix(F.avg_pool2d(h, 3, 1, 1))
        elif self.mode == "yolo":
            local = local + torch.roll(h, shifts=1, dims=-1)
        elif self.mode == "anchor":
            local = local + self.mix(h)
        elif self.mode == "cascade":
            local = local + self.mix(local)
        elif self.mode == "transformer":
            local = local * torch.sigmoid(self.mix(h)) + self.depthwise(h)
        elif self.mode == "occlusion":
            local = local + (h - F.avg_pool2d(h, 5, 1, 2))
        elif self.mode == "scale":
            local = local + F.interpolate(F.avg_pool2d(h, 2, 2), size=h.shape[-2:], mode="nearest")
        elif self.mode == "night":
            local = local + torch.tanh(self.mix(h))
        elif self.mode == "mamba":
            local = local + torch.tanh(self.depthwise(torch.roll(h, shifts=1, dims=-2)))
        return x + 0.2 * local


class TinyPedestrianDetector(nn.Module):
    def __init__(
        self, *, family: str, mode: str, in_channels: int, width: int, depth: int, num_queries: int
    ) -> None:
        super().__init__()
        self.family = str(family)
        self.mode = str(mode)
        self.stem = nn.Conv2d(int(in_channels), int(width), kernel_size=3, padding=1)
        self.blocks = nn.ModuleList(
            [
                TinyPedestrianBlock(channels=int(width), mode=str(mode))
                for _ in range(max(1, int(depth)))
            ]
        )
        self.query_embed = nn.Parameter(torch.randn(1, int(num_queries), int(width)) * 0.02)
        self.query_proj = nn.Linear(int(width), int(width))
        self.cls_head = nn.Linear(int(width), 1)
        self.box_head = nn.Linear(int(width), 4)
        self.quality_head = nn.Linear(int(width), 1)

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        image = check_nchw(image)
        feat = F.relu(self.stem(image), inplace=True)
        for block in self.blocks:
            feat = block(feat)
        pooled = F.adaptive_avg_pool2d(feat, 1).flatten(1)
        queries = self.query_embed.expand(image.shape[0], -1, -1) + self.query_proj(
            pooled
        ).unsqueeze(1)
        logits = self.cls_head(queries).squeeze(-1)
        boxes = torch.sigmoid(self.box_head(queries))
        quality = torch.sigmoid(self.quality_head(queries).squeeze(-1))
        return {"pred_logits": logits, "pred_boxes": boxes, "quality": quality}


def build_toy_pedestrian_detector(
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
    num_queries = int(spec.get("queries", 32))
    return TinyPedestrianDetector(
        family=str(family),
        mode=str(mode),
        in_channels=int(in_channels),
        width=width,
        depth=depth,
        num_queries=num_queries,
    )


def smoke_test_pedestrian_detector(builder, variant: str) -> None:
    model = builder(in_channels=3, variant=variant, width_mult=0.5)
    out = model(torch.randn(2, 3, 64, 64))
    print(variant, tuple(out["pred_logits"].shape), tuple(out["pred_boxes"].shape))
    print("ok")
