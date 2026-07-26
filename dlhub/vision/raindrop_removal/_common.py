from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def check_nchw(image: torch.Tensor) -> torch.Tensor:
    image = image.to(torch.float32)
    if image.ndim != 4:
        raise ValueError(f"Expected input shape (B, C, H, W), got {tuple(image.shape)}")
    return image


class TinyDropBlock(nn.Module):
    def __init__(self, *, channels: int, mode: str) -> None:
        super().__init__()
        self.mode = str(mode)
        self.norm = nn.GroupNorm(1, int(channels))
        self.conv1 = nn.Conv2d(int(channels), int(channels), kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(int(channels), int(channels), kernel_size=3, padding=1)
        self.mix = nn.Conv2d(int(channels), int(channels), kernel_size=1)
        self.depthwise = nn.Conv2d(
            int(channels), int(channels), kernel_size=5, padding=2, groups=int(channels)
        )
        if self.mode == "prompt":
            self.prompt = nn.Parameter(torch.zeros(1, int(channels), 1, 1))
        else:
            self.register_parameter("prompt", None)

    def forward(self, x: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        if self.prompt is not None:
            h = h + self.prompt
        local = self.conv2(F.relu(self.conv1(h), inplace=True))
        if self.mode in {"mask_guided", "streak_aware", "texture_refine", "context"}:
            local = local + self.depthwise(h)
        elif self.mode == "dual_branch":
            local = local + self.mix(guide)
        elif self.mode == "recurrent":
            local = local + 0.5 * self.mix(h)
        elif self.mode == "transformer":
            attn = torch.sigmoid(self.mix(h))
            local = local * attn + self.depthwise(h)
        elif self.mode == "frequency":
            low = F.avg_pool2d(h, kernel_size=3, stride=1, padding=1)
            local = local + (h - low)
        elif self.mode == "prompt":
            local = local + self.mix(guide)
        elif self.mode == "mamba":
            local = local + torch.tanh(self.depthwise(torch.roll(h, shifts=1, dims=-1)))
        return x + 0.2 * local


class TinyDropRemover(nn.Module):
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
            [TinyDropBlock(channels=int(width), mode=str(mode)) for _ in range(max(1, int(depth)))]
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(int(width), int(width), kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(width), int(in_channels), kernel_size=3, padding=1),
        )
        self.mask_head = nn.Conv2d(int(width), 1, kernel_size=1)

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        image = check_nchw(image)
        restored = image
        last_mask = None
        for _ in range(self.steps):
            feat = self.encoder(restored)
            guide = self.guide(restored)
            for block in self.blocks:
                feat = block(feat, guide)
            residual = self.decoder(feat)
            mask_logits = self.mask_head(feat)
            mask = torch.sigmoid(mask_logits)
            if self.mode == "frequency":
                smooth = F.avg_pool2d(restored, kernel_size=3, stride=1, padding=1)
                residual = 0.6 * residual + 0.4 * (restored - smooth)
            elif self.mode == "transformer":
                residual = residual * (0.5 + 0.5 * mask)
            elif self.mode == "dual_branch":
                residual = 0.5 * residual + 0.5 * guide[:, : residual.shape[1]]
            restored = torch.clamp(restored - residual * mask, -1.0, 1.0)
            last_mask = mask
        drop_layer = torch.clamp(image - restored, -1.0, 1.0)
        return {
            "restored": restored,
            "raindrop_layer": drop_layer,
            "raindrop_mask": last_mask if last_mask is not None else torch.zeros_like(image[:, :1]),
        }


def build_toy_drop_remover(
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
    width = max(12, int(int(spec["width"]) * float(width_mult)))
    depth = int(spec["depth"])
    steps = int(spec.get("steps", 1))
    return TinyDropRemover(
        family=str(family),
        mode=str(mode),
        in_channels=int(in_channels),
        width=width,
        depth=depth,
        steps=steps,
    )


def smoke_test_drop_remover(builder, variant: str) -> None:
    model = builder(in_channels=3, variant=variant, width_mult=0.5)
    image = torch.randn(2, 3, 32, 32)
    out = model(image)
    print(variant, tuple(out["restored"].shape), tuple(out["raindrop_mask"].shape))
    print("ok")
