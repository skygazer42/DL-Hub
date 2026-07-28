from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def check_nchw(image: torch.Tensor) -> torch.Tensor:
    image = image.to(torch.float32)
    if image.ndim != 4:
        raise ValueError(f"Expected input shape (B, C, H, W), got {tuple(image.shape)}")
    return image


class TinyFewShotBlock(nn.Module):
    def __init__(self, *, channels: int, mode: str) -> None:
        super().__init__()
        self.mode = str(mode)
        self.norm = nn.GroupNorm(1, int(channels))
        self.conv1 = nn.Conv2d(int(channels), int(channels), 3, padding=1)
        self.conv2 = nn.Conv2d(int(channels), int(channels), 3, padding=1)
        self.mix = nn.Conv2d(int(channels), int(channels), 1)
        self.depthwise = nn.Conv2d(
            int(channels),
            int(channels),
            5,
            padding=2,
            groups=int(channels),
        )
        self.prompt = (
            nn.Parameter(torch.zeros(1, int(channels), 1, 1)) if self.mode == "prompt" else None
        )

    def forward(self, x: torch.Tensor, support_feat: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        if self.prompt is not None:
            h = h + self.prompt
        local = self.conv2(F.relu(self.conv1(h), inplace=True))
        if self.mode in {"prototype", "matching", "relation", "attention", "hypercorr"}:
            local = local + self.depthwise(h)
        elif self.mode == "transformer":
            local = local * torch.sigmoid(self.mix(h)) + self.depthwise(h)
        elif self.mode == "dual":
            local = local + self.mix(support_feat)
        elif self.mode == "iterative":
            local = local + self.mix(F.avg_pool2d(h, 3, 1, 1))
        elif self.mode == "mamba":
            local = local + torch.tanh(self.depthwise(torch.roll(h, shifts=1, dims=-1)))
        return x + 0.2 * local


class TinyFewShotSegmentor(nn.Module):
    def __init__(
        self,
        *,
        family: str,
        mode: str,
        in_channels: int,
        width: int,
        depth: int,
        classes: int = 2,
    ) -> None:
        super().__init__()
        self.family = str(family)
        self.mode = str(mode)
        self.query_stem = nn.Conv2d(int(in_channels), int(width), 3, padding=1)
        self.support_stem = nn.Conv2d(int(in_channels), int(width), 3, padding=1)
        self.blocks = nn.ModuleList(
            [
                TinyFewShotBlock(channels=int(width), mode=str(mode))
                for _ in range(max(1, int(depth)))
            ]
        )
        self.logits = nn.Conv2d(int(width), int(classes), 1)
        self.aux = nn.Conv2d(int(width), 1, 1)

    def forward(
        self,
        query_image: torch.Tensor,
        support_image: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        query = check_nchw(query_image)
        support = query if support_image is None else check_nchw(support_image)

        q_feat = F.relu(self.query_stem(query), inplace=True)
        s_feat = F.relu(self.support_stem(support), inplace=True)
        proto = F.adaptive_avg_pool2d(s_feat, 1)
        proto_map = proto.expand(-1, -1, q_feat.shape[-2], q_feat.shape[-1])

        feat = q_feat + 0.1 * proto_map
        for block in self.blocks:
            feat = block(feat, proto_map)

        logits = self.logits(feat)
        return {
            "logits": logits,
            "mask": torch.softmax(logits, dim=1),
            "prototype_map": proto_map,
            "support_similarity": torch.sigmoid(self.aux(feat)),
        }


def build_baseline_few_shot_segmentor(
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
    classes = int(spec.get("classes", 2))
    return TinyFewShotSegmentor(
        family=str(family),
        mode=str(mode),
        in_channels=int(in_channels),
        width=width,
        depth=depth,
        classes=classes,
    )


def smoke_test_few_shot_segmentor(builder, variant: str) -> None:
    model = builder(in_channels=3, variant=variant, width_mult=0.5)
    query = torch.randn(2, 3, 64, 64)
    out = model(query)
    print(variant, tuple(out["logits"].shape), tuple(out["prototype_map"].shape))
    print("ok")
