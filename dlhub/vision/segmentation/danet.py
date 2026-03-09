
import torch
from torch import nn
import torch.nn.functional as F

from dlhub.vision.backbones._blocks import ConvBNAct, scale_channels
from dlhub.vision.segmentation._common import BackboneC2C3C4C5, check_nchw


class _PositionAttention(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        c = int(channels)
        mid = max(8, c // 8)
        self.q = nn.Conv2d(c, mid, kernel_size=1, bias=True)
        self.k = nn.Conv2d(c, mid, kernel_size=1, bias=True)
        self.v = nn.Conv2d(c, c, kernel_size=1, bias=True)
        self.proj = nn.Conv2d(c, c, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        q = self.q(x).flatten(2)  # (B, mid, N)
        k = self.k(x).flatten(2)  # (B, mid, N)
        v = self.v(x).flatten(2)  # (B, c, N)
        att = torch.softmax(torch.bmm(q.transpose(1, 2), k), dim=-1)  # (B, N, N)
        y = torch.bmm(v, att.transpose(1, 2)).view(b, c, h, w)
        return x + self.proj(y)


class _ChannelAttention(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        c = int(channels)
        self.proj = nn.Conv2d(c, c, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        f = x.flatten(2)  # (B, C, N)
        att = torch.softmax(torch.bmm(f, f.transpose(1, 2)), dim=-1)  # (B, C, C)
        y = torch.bmm(att, f).view(b, c, h, w)
        return x + self.proj(y)


class DANet(nn.Module):
    """DANet (Dual Attention Network) semantic segmentation (toy-first)."""

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
        feat_channels: int = 64,
    ) -> None:
        super().__init__()
        nc = int(num_classes)
        if nc <= 0:
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

        self.proj = ConvBNAct(int(c3_channels), fc, kernel_size=1, stride=1, padding=0, act="relu")
        self.pa = _PositionAttention(fc)
        self.ca = _ChannelAttention(fc)
        self.fuse = nn.Sequential(
            ConvBNAct(fc, fc, kernel_size=3, stride=1, act="relu"),
            nn.Conv2d(fc, nc, kernel_size=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = check_nchw(x)
        inp_hw = x.shape[-2:]
        _, c3, _, _ = self.backbone(x)  # stride /8
        f = self.proj(c3)
        f = self.pa(f)
        f = self.ca(f)
        logits8 = self.fuse(f)
        return F.interpolate(logits8, size=inp_hw, mode="nearest")


_VARIANTS: dict[str, dict] = {
    "danet_tiny": {"stem": 24, "c2": 24, "c3": 48, "c4": 64, "c5": 96, "depth": 1, "feat": 48},
    "danet_small": {"stem": 24, "c2": 32, "c3": 64, "c4": 96, "c5": 128, "depth": 2, "feat": 64},
    "danet_base": {"stem": 32, "c2": 40, "c3": 80, "c4": 128, "c5": 160, "depth": 2, "feat": 80},
}


def build_danet_segmenter(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "danet_small",
    width_mult: float = 1.0,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown DANet variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]

    stem = scale_channels(int(spec["stem"]), float(width_mult), min_ch=16, divisor=8)
    c2 = scale_channels(int(spec["c2"]), float(width_mult), min_ch=16, divisor=8)
    c3 = scale_channels(int(spec["c3"]), float(width_mult), min_ch=16, divisor=8)
    c4 = scale_channels(int(spec["c4"]), float(width_mult), min_ch=16, divisor=8)
    c5 = scale_channels(int(spec["c5"]), float(width_mult), min_ch=16, divisor=8)
    feat = scale_channels(int(spec["feat"]), float(width_mult), min_ch=16, divisor=8)

    return DANet(
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
    m = build_danet_segmenter(in_channels=3, num_classes=4, variant="danet_tiny", width_mult=0.5)
    y = m(x)
    print("danet_tiny", tuple(y.shape))
    loss = y.mean()
    loss.backward()
    print("ok")

