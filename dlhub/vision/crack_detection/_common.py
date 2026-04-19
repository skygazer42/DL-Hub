from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def check_nchw(image: torch.Tensor) -> torch.Tensor:
    image = image.to(torch.float32)
    if image.ndim != 4:
        raise ValueError(f"Expected input shape (B, C, H, W), got {tuple(image.shape)}")
    return image


class TinyCrackBlock(nn.Module):
    def __init__(self, *, channels: int, mode: str) -> None:
        super().__init__()
        self.mode = str(mode)
        self.norm = nn.GroupNorm(1, int(channels))
        self.conv1 = nn.Conv2d(int(channels), int(channels), 3, padding=1)
        self.conv2 = nn.Conv2d(int(channels), int(channels), 3, padding=1)
        self.mix = nn.Conv2d(int(channels), int(channels), 1)
        self.depthwise = nn.Conv2d(int(channels), int(channels), 5, padding=2, groups=int(channels))
        self.prompt = nn.Parameter(torch.zeros(1, int(channels), 1, 1)) if self.mode == "prompt" else None

    def forward(self, x: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        if self.prompt is not None:
            h = h + self.prompt
        local = self.conv2(F.relu(self.conv1(h), inplace=True))
        if self.mode in {"unet", "hed", "fpn", "contour", "skeleton"}:
            local = local + self.depthwise(h)
        elif self.mode == "transformer":
            local = local * torch.sigmoid(self.mix(h)) + self.depthwise(h)
        elif self.mode == "dual":
            local = local + self.mix(guide)
        elif self.mode == "coarse":
            local = local + self.mix(F.avg_pool2d(h, 3, 1, 1))
        elif self.mode == "mamba":
            local = local + torch.tanh(self.depthwise(torch.roll(h, shifts=1, dims=-1)))
        return x + 0.2 * local


class TinyCrackDetector(nn.Module):
    def __init__(
        self,
        *,
        family: str,
        mode: str,
        in_channels: int,
        width: int,
        depth: int,
        classes: int,
    ) -> None:
        super().__init__()
        self.family = str(family)
        self.mode = str(mode)
        self.stem = nn.Conv2d(int(in_channels), int(width), 3, padding=1)
        self.guide = nn.Conv2d(int(in_channels), int(width), 1)
        self.blocks = nn.ModuleList(
            [TinyCrackBlock(channels=int(width), mode=str(mode)) for _ in range(max(1, int(depth)))]
        )
        self.logit_head = nn.Conv2d(int(width), int(classes), 1)
        self.boundary_head = nn.Conv2d(int(width), 1, 1)
        self.thinness_head = nn.Conv2d(int(width), 1, 1)
        self.score_head = nn.Linear(int(width), 1)

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        image = check_nchw(image)
        gray = image.mean(dim=1, keepdim=True)
        grad_x = gray[..., 1:] - gray[..., :-1]
        grad_y = gray[..., 1:, :] - gray[..., :-1, :]
        grad_x = F.pad(grad_x, (0, 1, 0, 0))
        grad_y = F.pad(grad_y, (0, 0, 0, 1))
        guide = self.guide(torch.sqrt(1e-6 + grad_x.square() + grad_y.square()).expand_as(image))

        feat = F.relu(self.stem(image), inplace=True)
        for block in self.blocks:
            feat = block(feat, guide)
        logits = self.logit_head(feat)
        pooled = F.adaptive_avg_pool2d(feat, 1).flatten(1)
        return {
            "logits": logits,
            "mask": torch.softmax(logits, dim=1),
            "boundary": torch.sigmoid(self.boundary_head(feat)),
            "thinness": torch.sigmoid(self.thinness_head(feat)),
            "confidence": torch.sigmoid(self.score_head(pooled)),
        }


def build_toy_crack_detector(
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
        raise ValueError(f"Unknown variant for {family}: {variant!r}. Available: {sorted(variants)}")
    spec = dict(variants[name])
    width = max(16, int(int(spec["width"]) * float(width_mult)))
    depth = int(spec["depth"])
    classes = int(spec.get("classes", 2))
    return TinyCrackDetector(
        family=str(family),
        mode=str(mode),
        in_channels=int(in_channels),
        width=width,
        depth=depth,
        classes=classes,
    )


def smoke_test_crack_detector(builder, variant: str) -> None:
    model = builder(in_channels=3, variant=variant, width_mult=0.5)
    out = model(torch.randn(2, 3, 64, 64))
    print(variant, tuple(out["logits"].shape), tuple(out["boundary"].shape))
    print("ok")
