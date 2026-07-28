import torch
from torch import nn

from dlhub.vision.backbones._blocks import ConvBNAct, scale_channels


class _SSDBackbone(nn.Module):
    """Tiny SSD-style backbone producing multi-scale feature maps.

    Returns: (C3, C4, C5) with strides roughly (/8, /16, /32).
    """

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
            layers.append(
                ConvBNAct(in_ch, out_ch, kernel_size=3, stride=2 if down else 1, act="relu")
            )
            for _ in range(d - 1):
                layers.append(ConvBNAct(out_ch, out_ch, kernel_size=3, stride=1, act="relu"))
            return nn.Sequential(*layers)

        self.stage3 = stage(stem, int(c3_channels), down=True)  # /8
        self.stage4 = stage(int(c3_channels), int(c4_channels), down=True)  # /16
        self.stage5 = stage(int(c4_channels), int(c5_channels), down=True)  # /32

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        c3 = self.stage3(x)
        c4 = self.stage4(c3)
        c5 = self.stage5(c4)
        return c3, c4, c5


class SSDHead(nn.Module):
    """SSD-style conv head (per-level)."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        num_anchors: int,
        num_convs: int = 2,
    ) -> None:
        super().__init__()
        c = int(in_channels)
        nc = int(num_classes)
        na = int(num_anchors)
        n = int(num_convs)
        if nc <= 0:
            raise ValueError("num_classes must be > 0")
        if na <= 0:
            raise ValueError("num_anchors must be > 0")
        if n <= 0:
            raise ValueError("num_convs must be > 0")

        tower: list[nn.Module] = []
        for _ in range(n):
            tower.append(ConvBNAct(c, c, kernel_size=3, stride=1, act="relu"))
        self.tower = nn.Sequential(*tower)
        self.cls = nn.Conv2d(c, na * nc, kernel_size=3, padding=1)
        self.box = nn.Conv2d(c, na * 4, kernel_size=3, padding=1)

    def forward_single(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        f = self.tower(x)
        return self.cls(f), self.box(f)

    def forward(
        self, feats: tuple[torch.Tensor, ...]
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        cls_out: list[torch.Tensor] = []
        box_out: list[torch.Tensor] = []
        for f in feats:
            c, b = self.forward_single(f)
            cls_out.append(c)
            box_out.append(b)
        return cls_out, box_out


class SSDDetector(nn.Module):
    """SSD-style single-stage detector (compact-first).

    Forward returns raw outputs (no decoding):
    - cls_logits: list[(B, A*C, Hi, Wi)]
    - bbox_deltas: list[(B, A*4, Hi, Wi)]
    """

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        stem_channels: int = 32,
        backbone_channels: tuple[int, int, int] = (64, 96, 128),
        backbone_depth: int = 2,
        num_anchors: int = 6,
        head_convs: int = 2,
    ) -> None:
        super().__init__()
        c3, c4, c5 = (int(x) for x in backbone_channels)
        self.backbone = _SSDBackbone(
            in_channels=int(in_channels),
            stem_channels=int(stem_channels),
            c3_channels=c3,
            c4_channels=c4,
            c5_channels=c5,
            depth=int(backbone_depth),
        )
        # SSD uses independent heads per feature level (channels differ between C3/C4/C5).
        self.heads = nn.ModuleList(
            [
                SSDHead(
                    in_channels=int(ch),
                    num_classes=int(num_classes),
                    num_anchors=int(num_anchors),
                    num_convs=int(head_convs),
                )
                for ch in (c3, c4, c5)
            ]
        )

    def forward(self, x: torch.Tensor) -> dict[str, list[torch.Tensor]]:
        x = x.to(torch.float32)
        if x.ndim != 4:
            raise ValueError(f"Expected input shape (B, C, H, W), got {tuple(x.shape)}")

        c3, c4, c5 = self.backbone(x)
        feats = (c3, c4, c5)
        cls_logits: list[torch.Tensor] = []
        bbox_deltas: list[torch.Tensor] = []
        for head, feat in zip(self.heads, feats, strict=True):
            c, b = head.forward_single(feat)
            cls_logits.append(c)
            bbox_deltas.append(b)
        return {"cls_logits": cls_logits, "bbox_deltas": bbox_deltas}


_VARIANTS: dict[str, dict] = {
    "ssd_tiny": {"stem": 24, "c3": 48, "c4": 64, "c5": 80, "depth": 1, "head": 1},
    "ssd_small": {"stem": 32, "c3": 64, "c4": 96, "c5": 128, "depth": 2, "head": 2},
    "ssd_base": {"stem": 48, "c3": 96, "c4": 144, "c5": 192, "depth": 3, "head": 3},
}


def build_ssd_detector(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "ssd_tiny",
    width_mult: float = 1.0,
    num_anchors: int = 6,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown SSD variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]

    stem = scale_channels(int(spec["stem"]), float(width_mult), min_ch=16, divisor=8)
    c3 = scale_channels(int(spec["c3"]), float(width_mult), min_ch=16, divisor=8)
    c4 = scale_channels(int(spec["c4"]), float(width_mult), min_ch=16, divisor=8)
    c5 = scale_channels(int(spec["c5"]), float(width_mult), min_ch=16, divisor=8)

    return SSDDetector(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        stem_channels=int(stem),
        backbone_channels=(int(c3), int(c4), int(c5)),
        backbone_depth=int(spec["depth"]),
        num_anchors=int(num_anchors),
        head_convs=int(spec["head"]),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 128, 128)
    m = build_ssd_detector(in_channels=3, num_classes=3, variant="ssd_tiny", width_mult=0.5)
    out = m(x)
    print(
        "ssd_tiny",
        [tuple(t.shape) for t in out["cls_logits"]],
        [tuple(t.shape) for t in out["bbox_deltas"]],
    )
    loss = sum(t.mean() for t in out["cls_logits"]) + sum(t.mean() for t in out["bbox_deltas"])
    loss.backward()
    print("ok")
