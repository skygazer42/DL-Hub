from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def check_nchw(image: torch.Tensor) -> torch.Tensor:
    image = image.to(torch.float32)
    if image.ndim != 4:
        raise ValueError(f"Expected input shape (B, C, H, W), got {tuple(image.shape)}")
    return image


class TinyCARBlock(nn.Module):
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
        self.prompt = nn.Parameter(torch.zeros(1, int(channels), 1, 1)) if self.mode == "prompt" else None

    def forward(self, x: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        if self.prompt is not None:
            h = h + self.prompt
        local = self.conv2(F.relu(self.conv1(h), inplace=True))
        if self.mode in {"jpeg", "residual", "iterative"}:
            local = local + self.depthwise(h)
        elif self.mode == "dct":
            low = F.avg_pool2d(h, kernel_size=3, stride=1, padding=1)
            local = local + (h - low)
        elif self.mode == "frequency":
            local = local + 0.5 * self.mix(guide)
        elif self.mode == "dual":
            local = local + self.mix(guide) + torch.roll(h, shifts=1, dims=-1)
        elif self.mode == "transformer":
            attn = torch.sigmoid(self.mix(h))
            local = local * attn + self.depthwise(h)
        elif self.mode == "diffusion":
            local = 0.7 * local + 0.3 * torch.tanh(self.mix(h))
        elif self.mode == "mamba":
            local = local + torch.tanh(self.depthwise(torch.roll(h, shifts=1, dims=-2)))
        return x + 0.2 * local


class TinyArtifactReducer(nn.Module):
    def __init__(
        self,
        *,
        family: str,
        mode: str,
        in_channels: int,
        width: int,
        depth: int,
        steps: int,
    ) -> None:
        super().__init__()
        self.family = str(family)
        self.mode = str(mode)
        self.steps = max(1, int(steps))
        self.encoder = nn.Conv2d(int(in_channels), int(width), kernel_size=3, padding=1)
        self.guide = nn.Conv2d(int(in_channels), int(width), kernel_size=1)
        self.blocks = nn.ModuleList(
            [TinyCARBlock(channels=int(width), mode=str(mode)) for _ in range(max(1, int(depth)))]
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(int(width), int(width), kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(width), int(in_channels), kernel_size=3, padding=1),
        )
        self.quality_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(int(width), 1),
        )

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        image = check_nchw(image)
        clean = image
        for _ in range(self.steps):
            feat = self.encoder(clean)
            guide = self.guide(clean)
            for block in self.blocks:
                feat = block(feat, guide)
            artifact = self.decoder(feat)
            if self.mode == "dct":
                smooth = F.avg_pool2d(clean, kernel_size=3, stride=1, padding=1)
                artifact = 0.6 * artifact + 0.4 * (clean - smooth)
            elif self.mode == "diffusion":
                artifact = 0.5 * artifact + 0.5 * (clean - torch.tanh(clean))
            clean = torch.clamp(image - artifact, -1.0, 1.0)
        artifact_layer = torch.clamp(image - clean, -1.0, 1.0)
        quality = torch.sigmoid(self.quality_head(self.encoder(clean)))
        return {"restored": clean, "artifact": artifact_layer, "quality": quality}


def build_toy_artifact_reducer(
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
    width = max(12, int(int(spec["width"]) * float(width_mult)))
    depth = int(spec["depth"])
    steps = int(spec.get("steps", 1))
    return TinyArtifactReducer(
        family=str(family),
        mode=str(mode),
        in_channels=int(in_channels),
        width=width,
        depth=depth,
        steps=steps,
    )


def smoke_test_artifact_reducer(builder, variant: str) -> None:
    model = builder(in_channels=3, variant=variant, width_mult=0.5)
    out = model(torch.randn(2, 3, 32, 32))
    print(variant, tuple(out["restored"].shape), tuple(out["quality"].shape))
    print("ok")

