import torch
import torch.nn.functional as F
from torch import nn

from dlhub.vision.backbones._blocks import ConvBNAct, scale_channels


class _BackboneC3C5(nn.Module):
    """Tiny backbone returning (C3, C4, C5)."""

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

        def stage(in_ch: int, out_ch: int) -> nn.Sequential:
            layers: list[nn.Module] = []
            layers.append(ConvBNAct(in_ch, out_ch, kernel_size=3, stride=2, act="relu"))
            for _ in range(d - 1):
                layers.append(ConvBNAct(out_ch, out_ch, kernel_size=3, stride=1, act="relu"))
            return nn.Sequential(*layers)

        self.c3 = stage(stem, int(c3_channels))  # /8
        self.c4 = stage(int(c3_channels), int(c4_channels))  # /16
        self.c5 = stage(int(c4_channels), int(c5_channels))  # /32

        self.lat4 = nn.Conv2d(int(c4_channels), int(c3_channels), kernel_size=1)
        self.lat5 = nn.Conv2d(int(c5_channels), int(c3_channels), kernel_size=1)
        self.smooth3 = nn.Conv2d(int(c3_channels), int(c3_channels), kernel_size=3, padding=1)
        self.smooth4 = nn.Conv2d(int(c3_channels), int(c3_channels), kernel_size=3, padding=1)
        self.smooth5 = nn.Conv2d(int(c3_channels), int(c3_channels), kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        c3 = self.c3(x)
        c4 = self.c4(c3)
        c5 = self.c5(c4)

        # DSSD-style fusion: deconv/top-down refinement (toy: nearest upsample + lateral add).
        p5 = self.lat5(c5)
        p4 = self.lat4(c4) + F.interpolate(p5, size=c4.shape[-2:], mode="nearest")
        p3 = c3 + F.interpolate(p4, size=c3.shape[-2:], mode="nearest")
        return self.smooth3(p3), self.smooth4(p4), self.smooth5(p5)


class DSSDHead(nn.Module):
    def __init__(
        self, *, channels: int, num_classes: int, num_anchors: int, num_convs: int = 2
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

        tower = nn.Sequential(
            *[ConvBNAct(c, c, kernel_size=3, stride=1, act="relu") for _ in range(n)]
        )
        self.cls_tower = tower
        self.box_tower = tower
        self.cls = nn.Conv2d(c, na * nc, kernel_size=3, padding=1)
        self.box = nn.Conv2d(c, na * 4, kernel_size=3, padding=1)

    def forward(
        self, feats: tuple[torch.Tensor, ...]
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        cls_out: list[torch.Tensor] = []
        box_out: list[torch.Tensor] = []
        for f in feats:
            cls_out.append(self.cls(self.cls_tower(f)))
            box_out.append(self.box(self.box_tower(f)))
        return cls_out, box_out


class DSSDDetector(nn.Module):
    """DSSD-style SSD with simple top-down refinement (toy-first)."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        stem_channels: int = 32,
        backbone_channels: tuple[int, int, int] = (64, 96, 128),
        backbone_depth: int = 2,
        feat_channels: int = 64,
        num_anchors: int = 6,
        head_convs: int = 2,
    ) -> None:
        super().__init__()
        c3, c4, c5 = (int(x) for x in backbone_channels)
        self.backbone = _BackboneC3C5(
            in_channels=int(in_channels),
            stem_channels=int(stem_channels),
            c3_channels=c3,
            c4_channels=c4,
            c5_channels=c5,
            depth=int(backbone_depth),
        )
        self.proj = nn.Conv2d(int(c3), int(feat_channels), kernel_size=1)
        self.head = DSSDHead(
            channels=int(feat_channels),
            num_classes=int(num_classes),
            num_anchors=int(num_anchors),
            num_convs=int(head_convs),
        )

    def forward(self, x: torch.Tensor) -> dict[str, list[torch.Tensor]]:
        x = x.to(torch.float32)
        if x.ndim != 4:
            raise ValueError(f"Expected input shape (B, C, H, W), got {tuple(x.shape)}")
        p3, p4, p5 = self.backbone(x)
        p3 = self.proj(p3)
        p4 = self.proj(p4)
        p5 = self.proj(p5)
        cls_logits, bbox_deltas = self.head((p3, p4, p5))
        return {"cls_logits": cls_logits, "bbox_deltas": bbox_deltas}


_VARIANTS: dict[str, dict] = {
    "dssd_tiny": {"stem": 24, "c3": 48, "c4": 64, "c5": 80, "depth": 1, "feat": 64, "head": 1},
    "dssd_small": {"stem": 32, "c3": 64, "c4": 96, "c5": 128, "depth": 2, "feat": 96, "head": 2},
    "dssd_base": {"stem": 48, "c3": 96, "c4": 144, "c5": 192, "depth": 3, "feat": 128, "head": 3},
}


def build_dssd_detector(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "dssd_tiny",
    width_mult: float = 1.0,
    num_anchors: int = 6,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown DSSD variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]

    stem = scale_channels(int(spec["stem"]), float(width_mult), min_ch=16, divisor=8)
    c3 = scale_channels(int(spec["c3"]), float(width_mult), min_ch=16, divisor=8)
    c4 = scale_channels(int(spec["c4"]), float(width_mult), min_ch=16, divisor=8)
    c5 = scale_channels(int(spec["c5"]), float(width_mult), min_ch=16, divisor=8)
    feat = scale_channels(int(spec["feat"]), float(width_mult), min_ch=16, divisor=8)

    return DSSDDetector(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        stem_channels=int(stem),
        backbone_channels=(int(c3), int(c4), int(c5)),
        backbone_depth=int(spec["depth"]),
        feat_channels=int(feat),
        num_anchors=int(num_anchors),
        head_convs=int(spec["head"]),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 128, 128)
    m = build_dssd_detector(in_channels=3, num_classes=3, variant="dssd_tiny", width_mult=0.5)
    out = m(x)
    print(
        "dssd_tiny",
        [tuple(t.shape) for t in out["cls_logits"]],
        [tuple(t.shape) for t in out["bbox_deltas"]],
    )
    loss = sum(t.mean() for t in out["cls_logits"]) + sum(t.mean() for t in out["bbox_deltas"])
    loss.backward()
    print("ok")
