from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def check_nchw(image: torch.Tensor) -> torch.Tensor:
    image = image.to(torch.float32)
    if image.ndim != 4:
        raise ValueError(f"Expected input shape (B, C, H, W), got {tuple(image.shape)}")
    return image


class TinyOutpaintBlock(nn.Module):
    def __init__(self, *, channels: int, mode: str) -> None:
        super().__init__()
        self.mode = str(mode)
        self.norm = nn.GroupNorm(1, int(channels))
        self.conv1 = nn.Conv2d(int(channels), int(channels), 3, padding=1)
        self.conv2 = nn.Conv2d(int(channels), int(channels), 3, padding=1)
        self.mix = nn.Conv2d(int(channels), int(channels), 1)
        self.depthwise = nn.Conv2d(
            int(channels), int(channels), 5, padding=2, groups=int(channels)
        )
        self.prompt = (
            nn.Parameter(torch.zeros(1, int(channels), 1, 1))
            if self.mode == "prompt"
            else None
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        if self.prompt is not None:
            h = h + self.prompt
        local = self.conv2(F.relu(self.conv1(h), inplace=True))
        if self.mode in {"cnn", "coarse", "patch", "context", "boundary"}:
            local = local + self.depthwise(h)
        elif self.mode == "transformer":
            local = local * torch.sigmoid(self.mix(h)) + self.depthwise(h)
        elif self.mode == "dual":
            local = local + self.mix(cond)
        elif self.mode == "diffusion":
            local = local + torch.tanh(self.mix(cond - h))
        elif self.mode == "mamba":
            local = local + torch.tanh(self.depthwise(torch.roll(h, shifts=1, dims=-1)))
        return x + 0.2 * local


class TinyOutpainter(nn.Module):
    def __init__(
        self,
        *,
        family: str,
        mode: str,
        in_channels: int,
        width: int,
        depth: int,
    ) -> None:
        super().__init__()
        self.family = str(family)
        self.mode = str(mode)
        self.encoder = nn.Conv2d(int(in_channels), int(width), 3, padding=1)
        self.cond = nn.Conv2d(int(in_channels), int(width), 1)
        self.blocks = nn.ModuleList(
            [TinyOutpaintBlock(channels=int(width), mode=str(mode)) for _ in range(max(1, int(depth)))]
        )
        self.out_head = nn.Sequential(
            nn.Conv2d(int(width), int(width), 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(width), int(in_channels), 3, padding=1),
        )
        self.mask_head = nn.Conv2d(int(width), 1, 1)
        self.context_head = nn.Conv2d(int(width), int(width), 1)

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        image = check_nchw(image)
        feat = F.relu(self.encoder(image), inplace=True)
        cond = self.cond(image)
        for block in self.blocks:
            feat = block(feat, cond)
        mask = torch.sigmoid(self.mask_head(feat))
        predicted = torch.tanh(self.out_head(feat))
        outpainted = image * (1.0 - mask) + predicted * mask
        return {
            "outpainted": outpainted,
            "outpaint_mask": mask,
            "residual": outpainted - image,
            "context": self.context_head(feat),
        }


def build_toy_outpainter(
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
    return TinyOutpainter(
        family=str(family),
        mode=str(mode),
        in_channels=int(in_channels),
        width=width,
        depth=depth,
    )


def smoke_test_outpainter(builder, variant: str) -> None:
    model = builder(in_channels=3, variant=variant, width_mult=0.5)
    out = model(torch.randn(2, 3, 64, 64))
    print(variant, tuple(out["outpainted"].shape), tuple(out["outpaint_mask"].shape))
    print("ok")
