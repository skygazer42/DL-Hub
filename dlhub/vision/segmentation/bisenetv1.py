from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from dlhub.vision.backbones._blocks import ConvBNAct, scale_channels
from dlhub.vision.segmentation._common import BackboneC2C3C4C5, check_nchw


class _SEGate(nn.Module):
    def __init__(self, channels: int, *, reduction: int = 8) -> None:
        super().__init__()
        c = int(channels)
        r = int(reduction)
        mid = max(4, c // max(1, r))
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Conv2d(c, mid, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, c, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = self.fc(self.pool(x))
        return x * g


class BiSeNetV1(nn.Module):
    """BiSeNetV1 semantic segmentation (toy-first)."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        stem_channels: int = 24,
        c2_channels: int = 32,
        c3_channels: int = 64,
        c4_channels: int = 96,
        c5_channels: int = 128,
        depth: int = 2,
        spatial_channels: int = 64,
        context_channels: int = 96,
        fusion_channels: int = 96,
    ) -> None:
        super().__init__()
        nc = int(num_classes)
        if nc <= 0:
            raise ValueError("num_classes must be > 0")

        # Spatial path to /8.
        sp = int(spatial_channels)
        self.spatial = nn.Sequential(
            ConvBNAct(int(in_channels), sp, kernel_size=3, stride=2, act="relu"),  # /2
            ConvBNAct(sp, sp, kernel_size=3, stride=2, act="relu"),  # /4
            ConvBNAct(sp, sp, kernel_size=3, stride=2, act="relu"),  # /8
        )

        # Context path (backbone) provides /16 and /32.
        self.backbone = BackboneC2C3C4C5(
            in_channels=int(in_channels),
            stem_channels=int(stem_channels),
            c2_channels=int(c2_channels),
            c3_channels=int(c3_channels),
            c4_channels=int(c4_channels),
            c5_channels=int(c5_channels),
            depth=int(depth),
            act="relu",
        )
        cp = int(context_channels)
        self.c4_proj = ConvBNAct(int(c4_channels), cp, kernel_size=1, stride=1, padding=0, act="relu")
        self.c5_proj = ConvBNAct(int(c5_channels), cp, kernel_size=1, stride=1, padding=0, act="relu")
        self.global_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(int(c5_channels), cp, kernel_size=1, bias=False),
            nn.BatchNorm2d(cp),
            nn.ReLU(inplace=True),
        )

        f = int(fusion_channels)
        self.fuse = nn.Sequential(
            ConvBNAct(sp + cp, f, kernel_size=3, stride=1, act="relu"),
            _SEGate(f),
            nn.Conv2d(f, nc, kernel_size=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = check_nchw(x)
        inp_hw = x.shape[-2:]

        sp = self.spatial(x)  # /8
        _, _, c4, c5 = self.backbone(x)  # /16, /32

        c4p = self.c4_proj(c4)
        c5p = self.c5_proj(c5)
        g = self.global_proj(c5)
        g = F.interpolate(g, size=c5p.shape[-2:], mode="nearest")
        c5p = c5p + g

        c5_up = F.interpolate(c5p, size=c4p.shape[-2:], mode="nearest")
        ctx16 = c4p + c5_up
        ctx8 = F.interpolate(ctx16, size=sp.shape[-2:], mode="nearest")

        logits8 = self.fuse(torch.cat([sp, ctx8], dim=1))
        return F.interpolate(logits8, size=inp_hw, mode="nearest")


_VARIANTS: dict[str, dict] = {
    "bisenetv1_tiny": {"stem": 24, "c2": 24, "c3": 48, "c4": 64, "c5": 96, "depth": 1, "sp": 48, "cp": 64, "fuse": 64},
    "bisenetv1_small": {"stem": 24, "c2": 32, "c3": 64, "c4": 96, "c5": 128, "depth": 2, "sp": 64, "cp": 96, "fuse": 96},
    "bisenetv1_base": {"stem": 32, "c2": 40, "c3": 80, "c4": 128, "c5": 160, "depth": 2, "sp": 80, "cp": 128, "fuse": 128},
}


def build_bisenetv1_segmenter(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "bisenetv1_small",
    width_mult: float = 1.0,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown BiSeNetV1 variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]

    stem = scale_channels(int(spec["stem"]), float(width_mult), min_ch=16, divisor=8)
    c2 = scale_channels(int(spec["c2"]), float(width_mult), min_ch=16, divisor=8)
    c3 = scale_channels(int(spec["c3"]), float(width_mult), min_ch=16, divisor=8)
    c4 = scale_channels(int(spec["c4"]), float(width_mult), min_ch=16, divisor=8)
    c5 = scale_channels(int(spec["c5"]), float(width_mult), min_ch=16, divisor=8)
    sp = scale_channels(int(spec["sp"]), float(width_mult), min_ch=16, divisor=8)
    cp = scale_channels(int(spec["cp"]), float(width_mult), min_ch=16, divisor=8)
    fuse = scale_channels(int(spec["fuse"]), float(width_mult), min_ch=16, divisor=8)

    return BiSeNetV1(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        stem_channels=int(stem),
        c2_channels=int(c2),
        c3_channels=int(c3),
        c4_channels=int(c4),
        c5_channels=int(c5),
        depth=int(spec["depth"]),
        spatial_channels=int(sp),
        context_channels=int(cp),
        fusion_channels=int(fuse),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_bisenetv1_segmenter(in_channels=3, num_classes=4, variant="bisenetv1_tiny", width_mult=0.5)
    y = m(x)
    print("bisenetv1_tiny", tuple(y.shape))
    loss = y.mean()
    loss.backward()
    print("ok")

