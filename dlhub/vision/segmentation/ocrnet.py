from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from dlhub.vision.backbones._blocks import ConvBNAct, scale_channels
from dlhub.vision.segmentation._common import BackboneC2C3C4C5, check_nchw


class OCRNet(nn.Module):
    """OCRNet-style semantic segmentation (toy-first).

    Uses a coarse classifier to compute soft object regions, then aggregates object context
    and refines per-pixel predictions.
    """

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
        feat_channels: int = 96,
    ) -> None:
        super().__init__()
        k = int(num_classes)
        if k <= 0:
            raise ValueError("num_classes must be > 0")
        fc = int(feat_channels)
        if fc <= 0:
            raise ValueError("feat_channels must be > 0")

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

        self.proj = ConvBNAct(int(c4_channels), fc, kernel_size=1, stride=1, padding=0, act="relu")
        self.coarse = nn.Conv2d(fc, k, kernel_size=1, bias=True)
        self.refine = nn.Sequential(
            ConvBNAct(fc * 2, fc, kernel_size=3, stride=1, act="relu"),
            nn.Conv2d(fc, k, kernel_size=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = check_nchw(x)
        inp_hw = x.shape[-2:]
        _, _, c4, _ = self.backbone(x)  # /16
        feat = self.proj(c4)
        coarse = self.coarse(feat)  # (B,K,H,W)

        b, c, h, w = feat.shape
        k = coarse.shape[1]
        n = h * w

        probs = torch.softmax(coarse, dim=1).view(b, k, n)  # (B,K,N)
        f = feat.view(b, c, n)  # (B,C,N)

        # Region features: (B,K,C)
        region = torch.bmm(probs, f.transpose(1, 2))
        # Context per pixel: (B,C,N)
        context = torch.bmm(region.transpose(1, 2), probs).view(b, c, h, w)

        logits = self.refine(torch.cat([feat, context], dim=1))
        return F.interpolate(logits, size=inp_hw, mode="nearest")


_VARIANTS: dict[str, dict] = {
    "ocrnet_tiny": {"stem": 24, "c2": 24, "c3": 48, "c4": 64, "c5": 96, "depth": 1, "feat": 64},
    "ocrnet_small": {"stem": 24, "c2": 32, "c3": 64, "c4": 96, "c5": 128, "depth": 2, "feat": 96},
    "ocrnet_base": {"stem": 32, "c2": 40, "c3": 80, "c4": 128, "c5": 160, "depth": 2, "feat": 128},
}


def build_ocrnet_segmenter(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "ocrnet_small",
    width_mult: float = 1.0,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown OCRNet variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]

    stem = scale_channels(int(spec["stem"]), float(width_mult), min_ch=16, divisor=8)
    c2 = scale_channels(int(spec["c2"]), float(width_mult), min_ch=16, divisor=8)
    c3 = scale_channels(int(spec["c3"]), float(width_mult), min_ch=16, divisor=8)
    c4 = scale_channels(int(spec["c4"]), float(width_mult), min_ch=16, divisor=8)
    c5 = scale_channels(int(spec["c5"]), float(width_mult), min_ch=16, divisor=8)
    feat = scale_channels(int(spec["feat"]), float(width_mult), min_ch=16, divisor=8)

    return OCRNet(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        stem_channels=int(stem),
        c2_channels=int(c2),
        c3_channels=int(c3),
        c4_channels=int(c4),
        c5_channels=int(c5),
        depth=int(spec["depth"]),
        feat_channels=int(feat),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_ocrnet_segmenter(in_channels=3, num_classes=4, variant="ocrnet_tiny", width_mult=0.5)
    y = m(x)
    print("ocrnet_tiny", tuple(y.shape))
    loss = y.mean()
    loss.backward()
    print("ok")

