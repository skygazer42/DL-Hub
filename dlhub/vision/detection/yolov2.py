import torch
from torch import nn

from dlhub.vision.backbones._blocks import ConvBNAct, scale_channels


def _reorg(x: torch.Tensor, *, stride: int = 2) -> torch.Tensor:
    """YOLOv2 passthrough "reorg" (space-to-depth).

    Converts (B, C, H, W) -> (B, C*s*s, H/s, W/s).
    """

    s = int(stride)
    if s <= 0:
        raise ValueError("stride must be > 0")
    b, c, h, w = x.shape
    if h % s != 0 or w % s != 0:
        raise ValueError(f"Spatial dims must be divisible by stride={s}, got H={h}, W={w}")
    x = x.view(b, c, h // s, s, w // s, s)
    x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
    return x.view(b, c * s * s, h // s, w // s)


class _BackboneC2C3(nn.Module):
    """Tiny backbone returning two feature maps for YOLOv2-style fusion.

    Returns:
      - c2: stride /8
      - c3: stride /16
    """

    def __init__(
        self,
        *,
        in_channels: int,
        stem_channels: int,
        c2_channels: int,
        c3_channels: int,
        depth: int,
    ) -> None:
        super().__init__()
        c_in = int(in_channels)
        stem = int(stem_channels)
        d = int(depth)
        if d <= 0:
            raise ValueError("depth must be > 0")

        self.stem = nn.Sequential(
            ConvBNAct(c_in, stem, kernel_size=3, stride=2, act="leaky"),  # /2
            ConvBNAct(stem, stem, kernel_size=3, stride=2, act="leaky"),  # /4
        )

        def stage(in_ch: int, out_ch: int) -> nn.Sequential:
            layers: list[nn.Module] = []
            layers.append(ConvBNAct(in_ch, out_ch, kernel_size=3, stride=2, act="leaky"))
            for _ in range(d - 1):
                layers.append(ConvBNAct(out_ch, out_ch, kernel_size=3, stride=1, act="leaky"))
            return nn.Sequential(*layers)

        self.stage2 = stage(stem, int(c2_channels))  # /8
        self.stage3 = stage(int(c2_channels), int(c3_channels))  # /16

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        c2 = self.stage2(x)
        c3 = self.stage3(c2)
        return c2, c3


class YOLOv2Detector(nn.Module):
    """YOLOv2-style single-stage detector with passthrough fusion (toy-first)."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        stem_channels: int = 32,
        backbone_channels: tuple[int, int] = (64, 96),
        backbone_depth: int = 2,
        head_channels: int = 128,
        num_anchors: int = 5,
        passthrough_channels: int = 32,
    ) -> None:
        super().__init__()
        nc = int(num_classes)
        if nc <= 0:
            raise ValueError("num_classes must be > 0")

        c2, c3 = (int(x) for x in backbone_channels)
        self.backbone = _BackboneC2C3(
            in_channels=int(in_channels),
            stem_channels=int(stem_channels),
            c2_channels=c2,
            c3_channels=c3,
            depth=int(backbone_depth),
        )

        pt = int(passthrough_channels)
        self.passthrough = nn.Sequential(
            nn.Conv2d(c2, pt, kernel_size=1, bias=False),
            nn.BatchNorm2d(pt),
            nn.LeakyReLU(0.1, inplace=True),
        )

        fused = int(head_channels)
        self.fuse = ConvBNAct(c3 + pt * 4, fused, kernel_size=3, stride=1, act="leaky")

        na = int(num_anchors)
        self.obj = nn.Conv2d(fused, na, kernel_size=1)
        self.cls = nn.Conv2d(fused, na * nc, kernel_size=1)
        self.box = nn.Conv2d(fused, na * 4, kernel_size=1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = x.to(torch.float32)
        if x.ndim != 4:
            raise ValueError(f"Expected input shape (B, C, H, W), got {tuple(x.shape)}")
        c2, c3 = self.backbone(x)
        p = _reorg(self.passthrough(c2), stride=2)  # /8 -> /16
        f = self.fuse(torch.cat([c3, p], dim=1))
        return {
            "obj_logits": self.obj(f),
            "cls_logits": self.cls(f),
            "bbox_deltas": self.box(f),
        }


_VARIANTS: dict[str, dict] = {
    "yolov2_tiny": {"stem": 24, "c2": 48, "c3": 72, "depth": 1, "head": 96, "pt": 24},
    "yolov2_small": {"stem": 32, "c2": 64, "c3": 96, "depth": 2, "head": 128, "pt": 32},
    "yolov2_base": {"stem": 48, "c2": 96, "c3": 144, "depth": 3, "head": 192, "pt": 48},
}


def build_yolov2_detector(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "yolov2_tiny",
    width_mult: float = 1.0,
    num_anchors: int = 5,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown YOLOv2 variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    stem = scale_channels(int(spec["stem"]), float(width_mult), min_ch=16, divisor=8)
    c2 = scale_channels(int(spec["c2"]), float(width_mult), min_ch=16, divisor=8)
    c3 = scale_channels(int(spec["c3"]), float(width_mult), min_ch=16, divisor=8)
    head = scale_channels(int(spec["head"]), float(width_mult), min_ch=16, divisor=8)
    pt = scale_channels(int(spec["pt"]), float(width_mult), min_ch=16, divisor=8)
    return YOLOv2Detector(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        stem_channels=int(stem),
        backbone_channels=(int(c2), int(c3)),
        backbone_depth=int(spec["depth"]),
        head_channels=int(head),
        num_anchors=int(num_anchors),
        passthrough_channels=int(pt),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 128, 128)
    m = build_yolov2_detector(in_channels=3, num_classes=3, variant="yolov2_tiny", width_mult=0.5)
    out = m(x)
    print("yolov2_tiny", {k: tuple(v.shape) for k, v in out.items()})
    loss = sum(v.mean() for v in out.values())
    loss.backward()
    print("ok")
