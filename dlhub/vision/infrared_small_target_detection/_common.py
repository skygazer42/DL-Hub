from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def check_nchw(image: torch.Tensor) -> torch.Tensor:
    image = image.to(torch.float32)
    if image.ndim != 4:
        raise ValueError(f"Expected input shape (B, C, H, W), got {tuple(image.shape)}")
    return image


class TinyIRSTDBlock(nn.Module):
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
        if self.mode in {"tophat", "scrg", "aag", "mtunet", "anchor"}:
            local = local + self.depthwise(h)
        elif self.mode == "transformer":
            local = local * torch.sigmoid(self.mix(h)) + self.depthwise(h)
        elif self.mode == "cascade":
            local = local + self.mix(local)
        elif self.mode == "pyramid":
            local = local + self.mix(F.avg_pool2d(h, 3, 1, 1))
        elif self.mode == "context":
            local = local + torch.roll(h, shifts=1, dims=-1)
        elif self.mode == "mamba":
            local = local + torch.tanh(self.depthwise(torch.roll(h, shifts=1, dims=-2)))
        return x + 0.2 * local


class TinyIRSTDDetector(nn.Module):
    def __init__(
        self,
        *,
        family: str,
        mode: str,
        in_channels: int,
        width: int,
        depth: int,
        num_queries: int = 16,
    ) -> None:
        super().__init__()
        self.family = str(family)
        self.mode = str(mode)
        self.stem = nn.Conv2d(int(in_channels), int(width), 3, padding=1)
        self.blocks = nn.ModuleList(
            [TinyIRSTDBlock(channels=int(width), mode=str(mode)) for _ in range(max(1, int(depth)))]
        )
        self.target_head = nn.Conv2d(int(width), 1, 1)
        self.conf_head = nn.Conv2d(int(width), 1, 1)
        self.query = nn.Parameter(torch.randn(1, int(num_queries), int(width)) * 0.02)
        self.qproj = nn.Linear(int(width), int(width))
        self.box = nn.Linear(int(width), 4)
        self.score = nn.Linear(int(width), 1)

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        image = check_nchw(image)
        feat = F.relu(self.stem(image), inplace=True)
        for block in self.blocks:
            feat = block(feat)

        target_logits = self.target_head(feat)
        confidence = torch.sigmoid(self.conf_head(feat))
        pooled = F.adaptive_avg_pool2d(feat, 1).flatten(1)
        q = self.query.expand(image.shape[0], -1, -1) + self.qproj(pooled).unsqueeze(1)
        pred_boxes = torch.sigmoid(self.box(q))
        scores = torch.sigmoid(self.score(q)).squeeze(-1)

        return {
            "target_map": torch.sigmoid(target_logits),
            "target_logits": target_logits,
            "confidence": confidence,
            "pred_boxes": pred_boxes,
            "scores": scores,
        }


def build_baseline_irstd_detector(
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
    queries = int(spec.get("queries", 16))
    return TinyIRSTDDetector(
        family=str(family),
        mode=str(mode),
        in_channels=int(in_channels),
        width=width,
        depth=depth,
        num_queries=queries,
    )


def smoke_test_irstd_detector(builder, variant: str) -> None:
    model = builder(in_channels=3, variant=variant, width_mult=0.5)
    out = model(torch.randn(2, 3, 64, 64))
    print(variant, tuple(out["target_map"].shape), tuple(out["pred_boxes"].shape))
    print("ok")
