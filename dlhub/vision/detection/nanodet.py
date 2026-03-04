from __future__ import annotations

import torch
from torch import nn

from dlhub.vision.backbones._blocks import DepthwiseSeparableConv, scale_channels
from dlhub.vision.detection._common import BackboneC3C5, FPN, check_nchw


class NanoDetHead(nn.Module):
    """NanoDet / GFL-style head (toy).

    Outputs per feature level:
    - cls_logits: (B, C, H, W)
    - dist_logits: (B, 4*(reg_max+1), H, W)
    """

    def __init__(self, *, channels: int, num_classes: int, reg_max: int = 7, num_convs: int = 2) -> None:
        super().__init__()
        c = int(channels)
        nc = int(num_classes)
        rm = int(reg_max)
        n = int(num_convs)
        if nc <= 0:
            raise ValueError("num_classes must be > 0")
        if rm <= 0:
            raise ValueError("reg_max must be > 0")
        if n <= 0:
            raise ValueError("num_convs must be > 0")

        self.cls_tower = nn.Sequential(*[DepthwiseSeparableConv(c, c, act="silu") for _ in range(n)])
        self.reg_tower = nn.Sequential(*[DepthwiseSeparableConv(c, c, act="silu") for _ in range(n)])
        self.cls = nn.Conv2d(c, nc, kernel_size=1)
        self.dist = nn.Conv2d(c, 4 * (rm + 1), kernel_size=1)
        self.reg_max = rm

    def forward_single(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.cls(self.cls_tower(x)), self.dist(self.reg_tower(x))


class NanoDetDetector(nn.Module):
    """NanoDet-style lightweight detector (toy-first)."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        stem_channels: int = 24,
        backbone_channels: tuple[int, int, int] = (48, 64, 80),
        backbone_depth: int = 1,
        fpn_channels: int = 64,
        head_convs: int = 2,
        reg_max: int = 7,
    ) -> None:
        super().__init__()
        c3, c4, c5 = (int(x) for x in backbone_channels)
        out = int(fpn_channels)
        self.backbone = BackboneC3C5(
            in_channels=int(in_channels),
            stem_channels=int(stem_channels),
            c3_channels=c3,
            c4_channels=c4,
            c5_channels=c5,
            depth=int(backbone_depth),
            act="silu",
        )
        self.fpn = FPN((c3, c4, c5), out, act="silu")
        self.head = NanoDetHead(channels=out, num_classes=int(num_classes), reg_max=int(reg_max), num_convs=int(head_convs))

    def forward(self, x: torch.Tensor) -> dict[str, list[torch.Tensor]]:
        x = check_nchw(x)
        c3, c4, c5 = self.backbone(x)
        p3, p4, p5 = self.fpn(c3, c4, c5)
        cls_out, dist_out = [], []
        for p in (p3, p4, p5):
            c, d = self.head.forward_single(p)
            cls_out.append(c)
            dist_out.append(d)
        return {"cls_logits": cls_out, "dist_logits": dist_out}


_VARIANTS: dict[str, dict] = {
    "nanodet_tiny": {"stem": 20, "c3": 40, "c4": 56, "c5": 72, "depth": 1, "fpn": 56, "head": 1, "reg_max": 7},
    "nanodet_small": {"stem": 24, "c3": 48, "c4": 64, "c5": 80, "depth": 2, "fpn": 64, "head": 2, "reg_max": 7},
    "nanodet_base": {"stem": 32, "c3": 64, "c4": 96, "c5": 128, "depth": 2, "fpn": 96, "head": 2, "reg_max": 7},
}


def build_nanodet_detector(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "nanodet_tiny",
    width_mult: float = 1.0,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown NanoDet variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    stem = scale_channels(int(spec["stem"]), float(width_mult), min_ch=16, divisor=8)
    c3 = scale_channels(int(spec["c3"]), float(width_mult), min_ch=16, divisor=8)
    c4 = scale_channels(int(spec["c4"]), float(width_mult), min_ch=16, divisor=8)
    c5 = scale_channels(int(spec["c5"]), float(width_mult), min_ch=16, divisor=8)
    fpn = scale_channels(int(spec["fpn"]), float(width_mult), min_ch=16, divisor=8)
    return NanoDetDetector(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        stem_channels=int(stem),
        backbone_channels=(int(c3), int(c4), int(c5)),
        backbone_depth=int(spec["depth"]),
        fpn_channels=int(fpn),
        head_convs=int(spec["head"]),
        reg_max=int(spec["reg_max"]),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 128, 128)
    m = build_nanodet_detector(in_channels=3, num_classes=2, variant="nanodet_tiny", width_mult=0.5)
    out = m(x)
    print("nanodet_tiny", [tuple(t.shape) for t in out["cls_logits"]], [tuple(t.shape) for t in out["dist_logits"]])
    loss = sum(t.mean() for t in out["cls_logits"]) + sum(t.mean() for t in out["dist_logits"])
    loss.backward()
    print("ok")

