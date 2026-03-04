from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from dlhub.vision.backbones._blocks import ConvBNAct, scale_channels


class _SimpleBackboneC3C5(nn.Module):
    """Tiny conv backbone that returns (C3, C4, C5) feature maps."""

    def __init__(
        self,
        *,
        in_channels: int,
        stem_channels: int,
        c3_channels: int,
        c4_channels: int,
        c5_channels: int,
        depth: int,
    ) -> None:
        super().__init__()
        c_in = int(in_channels)
        stem = int(stem_channels)
        d = int(depth)
        if d <= 0:
            raise ValueError("depth must be > 0")

        self.stem = nn.Sequential(
            ConvBNAct(c_in, stem, kernel_size=3, stride=2, act="relu"),  # /2
            ConvBNAct(stem, stem, kernel_size=3, stride=2, act="relu"),  # /4
        )

        def stage(in_ch: int, out_ch: int, *, down: bool) -> nn.Sequential:
            layers: list[nn.Module] = []
            layers.append(ConvBNAct(in_ch, out_ch, kernel_size=3, stride=2 if down else 1, act="relu"))
            for _ in range(d - 1):
                layers.append(ConvBNAct(out_ch, out_ch, kernel_size=3, stride=1, act="relu"))
            return nn.Sequential(*layers)

        # stride /8, /16, /32
        self.stage3 = stage(stem, int(c3_channels), down=True)
        self.stage4 = stage(int(c3_channels), int(c4_channels), down=True)
        self.stage5 = stage(int(c4_channels), int(c5_channels), down=True)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        c3 = self.stage3(x)
        c4 = self.stage4(c3)
        c5 = self.stage5(c4)
        return c3, c4, c5


class FPN(nn.Module):
    def __init__(self, in_channels: tuple[int, int, int], out_channels: int) -> None:
        super().__init__()
        c3, c4, c5 = (int(x) for x in in_channels)
        out = int(out_channels)
        self.l3 = nn.Conv2d(c3, out, kernel_size=1)
        self.l4 = nn.Conv2d(c4, out, kernel_size=1)
        self.l5 = nn.Conv2d(c5, out, kernel_size=1)
        self.p3 = nn.Conv2d(out, out, kernel_size=3, padding=1)
        self.p4 = nn.Conv2d(out, out, kernel_size=3, padding=1)
        self.p5 = nn.Conv2d(out, out, kernel_size=3, padding=1)

    def forward(self, c3: torch.Tensor, c4: torch.Tensor, c5: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        p5 = self.l5(c5)
        p4 = self.l4(c4) + F.interpolate(p5, size=c4.shape[-2:], mode="nearest")
        p3 = self.l3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="nearest")
        return self.p3(p3), self.p4(p4), self.p5(p5)


class RetinaHead(nn.Module):
    def __init__(
        self,
        *,
        channels: int,
        num_classes: int,
        num_anchors: int = 9,
        num_convs: int = 4,
        prior_prob: float = 0.01,
    ) -> None:
        super().__init__()
        c = int(channels)
        nc = int(num_classes)
        na = int(num_anchors)
        n = int(num_convs)
        if nc <= 0:
            raise ValueError("num_classes must be > 0")
        if na <= 0:
            raise ValueError("num_anchors must be > 0")
        if n <= 0:
            raise ValueError("num_convs must be > 0")

        def tower() -> nn.Sequential:
            return nn.Sequential(*[ConvBNAct(c, c, kernel_size=3, stride=1, act="relu") for _ in range(n)])

        self.cls_tower = tower()
        self.box_tower = tower()

        self.cls_logits = nn.Conv2d(c, na * nc, kernel_size=3, padding=1)
        self.bbox_pred = nn.Conv2d(c, na * 4, kernel_size=3, padding=1)

        # Focal-loss style prior bias init.
        bias = -math.log((1.0 - float(prior_prob)) / float(prior_prob))
        nn.init.constant_(self.cls_logits.bias, bias)

    def forward_single(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cls = self.cls_logits(self.cls_tower(x))
        box = self.bbox_pred(self.box_tower(x))
        return cls, box

    def forward(self, feats: tuple[torch.Tensor, ...]) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        cls_out: list[torch.Tensor] = []
        box_out: list[torch.Tensor] = []
        for f in feats:
            c, b = self.forward_single(f)
            cls_out.append(c)
            box_out.append(b)
        return cls_out, box_out


class RetinaNetDetector(nn.Module):
    """RetinaNet-style single-stage detector (toy-first).

    Forward returns raw outputs (no NMS / decoding):
    - cls_logits: list[Tensor] with shapes (B, A*C, Hi, Wi)
    - bbox_deltas: list[Tensor] with shapes (B, A*4, Hi, Wi)
    """

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        stem_channels: int = 32,
        backbone_channels: tuple[int, int, int] = (64, 96, 128),
        backbone_depth: int = 2,
        fpn_channels: int = 96,
        num_anchors: int = 9,
        head_convs: int = 4,
    ) -> None:
        super().__init__()
        c3, c4, c5 = (int(x) for x in backbone_channels)
        self.backbone = _SimpleBackboneC3C5(
            in_channels=int(in_channels),
            stem_channels=int(stem_channels),
            c3_channels=c3,
            c4_channels=c4,
            c5_channels=c5,
            depth=int(backbone_depth),
        )
        self.fpn = FPN((c3, c4, c5), int(fpn_channels))
        self.head = RetinaHead(
            channels=int(fpn_channels),
            num_classes=int(num_classes),
            num_anchors=int(num_anchors),
            num_convs=int(head_convs),
        )

    def forward(self, x: torch.Tensor) -> dict[str, list[torch.Tensor]]:
        x = x.to(torch.float32)
        if x.ndim != 4:
            raise ValueError(f"Expected input shape (B, C, H, W), got {tuple(x.shape)}")

        c3, c4, c5 = self.backbone(x)
        p3, p4, p5 = self.fpn(c3, c4, c5)
        cls_logits, bbox_deltas = self.head((p3, p4, p5))
        return {"cls_logits": cls_logits, "bbox_deltas": bbox_deltas}


_VARIANTS: dict[str, dict] = {
    "retinanet_tiny": {"stem": 24, "c3": 48, "c4": 64, "c5": 80, "depth": 1, "fpn": 64, "head": 2},
    "retinanet_small": {"stem": 32, "c3": 64, "c4": 96, "c5": 128, "depth": 2, "fpn": 96, "head": 4},
    "retinanet_base": {"stem": 48, "c3": 96, "c4": 144, "c5": 192, "depth": 3, "fpn": 128, "head": 4},
}


def build_retinanet_detector(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "retinanet_tiny",
    width_mult: float = 1.0,
    num_anchors: int = 9,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown RetinaNet variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]

    stem = scale_channels(int(spec["stem"]), float(width_mult), min_ch=16, divisor=8)
    c3 = scale_channels(int(spec["c3"]), float(width_mult), min_ch=16, divisor=8)
    c4 = scale_channels(int(spec["c4"]), float(width_mult), min_ch=16, divisor=8)
    c5 = scale_channels(int(spec["c5"]), float(width_mult), min_ch=16, divisor=8)
    fpn = scale_channels(int(spec["fpn"]), float(width_mult), min_ch=16, divisor=8)

    return RetinaNetDetector(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        stem_channels=int(stem),
        backbone_channels=(int(c3), int(c4), int(c5)),
        backbone_depth=int(spec["depth"]),
        fpn_channels=int(fpn),
        num_anchors=int(num_anchors),
        head_convs=int(spec["head"]),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 128, 128)
    m = build_retinanet_detector(in_channels=3, num_classes=5, variant="retinanet_tiny", width_mult=1.0)
    out = m(x)
    print("retinanet_tiny", [tuple(t.shape) for t in out["cls_logits"]], [tuple(t.shape) for t in out["bbox_deltas"]])
    loss = sum(t.mean() for t in out["cls_logits"]) + sum(t.mean() for t in out["bbox_deltas"])
    loss.backward()
    print("ok")

