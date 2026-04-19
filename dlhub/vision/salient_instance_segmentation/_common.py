from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def check_nchw(image: torch.Tensor) -> torch.Tensor:
    image = image.to(torch.float32)
    if image.ndim != 4:
        raise ValueError(f"Expected input shape (B, C, H, W), got {tuple(image.shape)}")
    return image


class TinySalientInstanceBlock(nn.Module):
    def __init__(self, *, channels: int, mode: str) -> None:
        super().__init__()
        self.mode = str(mode)
        self.norm = nn.GroupNorm(1, int(channels))
        self.conv1 = nn.Conv2d(int(channels), int(channels), 3, padding=1)
        self.conv2 = nn.Conv2d(int(channels), int(channels), 3, padding=1)
        self.mix = nn.Conv2d(int(channels), int(channels), 1)
        self.depthwise = nn.Conv2d(int(channels), int(channels), 5, padding=2, groups=int(channels))
        self.prompt = nn.Parameter(torch.zeros(1, int(channels), 1, 1)) if self.mode == "prompt" else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        if self.prompt is not None:
            h = h + self.prompt
        local = self.conv2(F.relu(self.conv1(h), inplace=True))
        if self.mode in {"topdown", "prototype", "query", "contour", "cascade"}:
            local = local + self.depthwise(h)
        elif self.mode == "transformer":
            local = local * torch.sigmoid(self.mix(h)) + self.depthwise(h)
        elif self.mode == "pyramid":
            local = local + self.mix(F.avg_pool2d(h, 3, 1, 1))
        elif self.mode == "dual":
            local = local + self.mix(torch.roll(h, shifts=1, dims=-1))
        elif self.mode == "mamba":
            local = local + torch.tanh(self.depthwise(torch.roll(h, shifts=1, dims=-2)))
        return x + 0.2 * local


class TinySalientInstanceSegmentor(nn.Module):
    def __init__(self, *, family: str, mode: str, in_channels: int, width: int, depth: int, num_queries: int = 16, num_protos: int = 8) -> None:
        super().__init__()
        self.family = str(family)
        self.mode = str(mode)
        self.stem = nn.Conv2d(int(in_channels), int(width), 3, padding=1)
        self.blocks = nn.ModuleList([TinySalientInstanceBlock(channels=int(width), mode=str(mode)) for _ in range(max(1, int(depth)))])
        self.query = nn.Parameter(torch.randn(1, int(num_queries), int(width)) * 0.02)
        self.proj = nn.Linear(int(width), int(width))
        self.saliency_head = nn.Linear(int(width), 1)
        self.box_head = nn.Linear(int(width), 4)
        self.coeff_head = nn.Linear(int(width), int(num_protos))
        self.proto_head = nn.Conv2d(int(width), int(num_protos), 1)

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        image = check_nchw(image)
        feat = F.relu(self.stem(image), inplace=True)
        for block in self.blocks:
            feat = block(feat)
        pooled = F.adaptive_avg_pool2d(feat, 1).flatten(1)
        q = self.query.expand(image.shape[0], -1, -1) + self.proj(pooled).unsqueeze(1)
        return {
            "pred_boxes": torch.sigmoid(self.box_head(q)),
            "saliency_logits": self.saliency_head(q).squeeze(-1),
            "mask_coeffs": self.coeff_head(q),
            "proto_masks": self.proto_head(feat),
        }


def build_toy_salient_instance_segmentor(*, family: str, mode: str, variants: dict[str, dict[str, int]], in_channels: int, variant: str, width_mult: float = 1.0) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in variants:
        raise ValueError(f"Unknown variant for {family}: {variant!r}. Available: {sorted(variants)}")
    spec = dict(variants[name])
    width = max(16, int(int(spec["width"]) * float(width_mult)))
    depth = int(spec["depth"])
    num_queries = int(spec.get("queries", 16))
    num_protos = int(spec.get("protos", 8))
    return TinySalientInstanceSegmentor(family=str(family), mode=str(mode), in_channels=int(in_channels), width=width, depth=depth, num_queries=num_queries, num_protos=num_protos)


def smoke_test_salient_instance_segmentor(builder, variant: str) -> None:
    model = builder(in_channels=3, variant=variant, width_mult=0.5)
    out = model(torch.randn(2, 3, 64, 64))
    print(variant, tuple(out["pred_boxes"].shape), tuple(out["proto_masks"].shape))
    print("ok")
