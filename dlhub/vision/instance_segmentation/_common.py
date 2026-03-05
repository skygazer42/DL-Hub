from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from dlhub.vision.backbones._blocks import ConvBNAct


def check_nchw(x: torch.Tensor) -> torch.Tensor:
    x = x.to(torch.float32)
    if x.ndim != 4:
        raise ValueError(f"Expected input shape (B, C, H, W), got {tuple(x.shape)}")
    return x


class ConvTower(nn.Module):
    """A small stack of Conv-BN-Act blocks."""

    def __init__(self, channels: int, *, num_convs: int = 2, act: str = "relu") -> None:
        super().__init__()
        c = int(channels)
        n = int(num_convs)
        if c <= 0:
            raise ValueError("channels must be > 0")
        if n <= 0:
            raise ValueError("num_convs must be > 0")
        self.net = nn.Sequential(*[ConvBNAct(c, c, kernel_size=3, stride=1, act=act) for _ in range(n)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BackboneLowDet(nn.Module):
    """Backbone that returns (low_stride4, det_stride8) feature maps."""

    def __init__(
        self,
        *,
        in_channels: int,
        stem_channels: int,
        low_channels: int,
        det_channels: int,
        depth: int,
        act: str = "relu",
    ) -> None:
        super().__init__()
        c_in = int(in_channels)
        stem = int(stem_channels)
        low = int(low_channels)
        det = int(det_channels)
        d = int(depth)
        if d <= 0:
            raise ValueError("depth must be > 0")

        self.stem = ConvBNAct(c_in, stem, kernel_size=3, stride=2, act=act)  # /2
        self.to_low = ConvBNAct(stem, low, kernel_size=3, stride=2, act=act)  # /4

        layers: list[nn.Module] = [ConvBNAct(low, det, kernel_size=3, stride=2, act=act)]  # /8
        for _ in range(d - 1):
            layers.append(ConvBNAct(det, det, kernel_size=3, stride=1, act=act))
        self.det = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        low = self.to_low(x)
        det = self.det(low)
        return low, det


class ProtoNet(nn.Module):
    """Prototype mask generator used by YOLACT/CondInst/BlendMask-style models."""

    def __init__(self, in_ch: int, proto_ch: int, *, depth: int = 3, act: str = "relu") -> None:
        super().__init__()
        c_in = int(in_ch)
        p = int(proto_ch)
        d = int(depth)
        if p <= 0:
            raise ValueError("proto_ch must be > 0")
        if d <= 0:
            raise ValueError("depth must be > 0")

        layers: list[nn.Module] = []
        ch = c_in
        for _ in range(d - 1):
            layers.append(ConvBNAct(ch, ch, kernel_size=3, stride=1, act=act))
        layers.append(nn.Conv2d(ch, p, kernel_size=1, bias=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DensePredHead(nn.Module):
    """Dense prediction head for one-stage instance segmentation (toy-first)."""

    def __init__(
        self,
        *,
        in_ch: int,
        num_classes: int,
        num_anchors: int,
        num_protos: int,
        head_channels: int,
        num_convs: int = 2,
        act: str = "relu",
    ) -> None:
        super().__init__()
        c_in = int(in_ch)
        nc = int(num_classes)
        na = int(num_anchors)
        np = int(num_protos)
        hc = int(head_channels)
        n = int(num_convs)
        if nc <= 0:
            raise ValueError("num_classes must be > 0")
        if na <= 0:
            raise ValueError("num_anchors must be > 0")
        if np <= 0:
            raise ValueError("num_protos must be > 0")
        if n <= 0:
            raise ValueError("num_convs must be > 0")

        layers: list[nn.Module] = [ConvBNAct(c_in, hc, kernel_size=3, stride=1, act=act)]
        for _ in range(n - 1):
            layers.append(ConvBNAct(hc, hc, kernel_size=3, stride=1, act=act))
        self.tower = nn.Sequential(*layers)

        self.cls_logits = nn.Conv2d(hc, na * nc, kernel_size=3, padding=1)
        self.box_pred = nn.Conv2d(hc, na * 4, kernel_size=3, padding=1)
        self.mask_coeff = nn.Conv2d(hc, na * np, kernel_size=3, padding=1)

        nn.init.normal_(self.cls_logits.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.cls_logits.bias, 0.0)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        t = self.tower(x)
        return self.cls_logits(t), self.box_pred(t), self.mask_coeff(t)


def upsample_like(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    if x.shape[-2:] == ref.shape[-2:]:
        return x
    return F.interpolate(x, size=ref.shape[-2:], mode="nearest")
