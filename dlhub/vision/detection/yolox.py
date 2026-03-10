import torch
from torch import nn

from dlhub.vision.backbones._blocks import ConvBNAct, scale_channels
from dlhub.vision.detection._common import FPN, BackboneC3C5, ConvTower, check_nchw


class YOLOXHead(nn.Module):
    """YOLOX-style decoupled head (shared across FPN levels)."""

    def __init__(
        self, *, channels: int, num_classes: int, num_anchors: int = 1, num_convs: int = 2
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

        self.stem = ConvBNAct(c, c, kernel_size=1, stride=1, padding=0, act="silu")
        self.cls_tower = ConvTower(c, num_convs=n, act="silu")
        self.reg_tower = ConvTower(c, num_convs=n, act="silu")

        self.obj = nn.Conv2d(c, na, kernel_size=1)
        self.cls = nn.Conv2d(c, na * nc, kernel_size=1)
        self.box = nn.Conv2d(c, na * 4, kernel_size=1)

    def forward_single(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        cls_feat = self.cls_tower(x)
        reg_feat = self.reg_tower(x)
        return self.obj(reg_feat), self.cls(cls_feat), self.box(reg_feat)


class YOLOXDetector(nn.Module):
    """YOLOX-style detector with a shared decoupled head (toy-first)."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        stem_channels: int = 32,
        backbone_channels: tuple[int, int, int] = (64, 96, 128),
        backbone_depth: int = 2,
        fpn_channels: int = 96,
        head_convs: int = 2,
        num_anchors: int = 1,
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
        self.head = YOLOXHead(
            channels=out,
            num_classes=int(num_classes),
            num_anchors=int(num_anchors),
            num_convs=int(head_convs),
        )

    def forward(self, x: torch.Tensor) -> dict[str, list[torch.Tensor]]:
        x = check_nchw(x)
        c3, c4, c5 = self.backbone(x)
        p3, p4, p5 = self.fpn(c3, c4, c5)
        obj, cls, box = [], [], []
        for p in (p3, p4, p5):
            o, c, b = self.head.forward_single(p)
            obj.append(o)
            cls.append(c)
            box.append(b)
        return {"obj_logits": obj, "cls_logits": cls, "bbox_deltas": box}


_VARIANTS: dict[str, dict] = {
    "yolox_tiny": {"stem": 24, "c3": 48, "c4": 64, "c5": 80, "depth": 1, "fpn": 64, "head": 1},
    "yolox_small": {"stem": 32, "c3": 64, "c4": 96, "c5": 128, "depth": 2, "fpn": 96, "head": 2},
    "yolox_base": {"stem": 48, "c3": 96, "c4": 144, "c5": 192, "depth": 3, "fpn": 128, "head": 2},
}


def build_yolox_detector(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "yolox_tiny",
    width_mult: float = 1.0,
    num_anchors: int = 1,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown YOLOX variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]

    stem = scale_channels(int(spec["stem"]), float(width_mult), min_ch=16, divisor=8)
    c3 = scale_channels(int(spec["c3"]), float(width_mult), min_ch=16, divisor=8)
    c4 = scale_channels(int(spec["c4"]), float(width_mult), min_ch=16, divisor=8)
    c5 = scale_channels(int(spec["c5"]), float(width_mult), min_ch=16, divisor=8)
    fpn = scale_channels(int(spec["fpn"]), float(width_mult), min_ch=16, divisor=8)

    return YOLOXDetector(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        stem_channels=int(stem),
        backbone_channels=(int(c3), int(c4), int(c5)),
        backbone_depth=int(spec["depth"]),
        fpn_channels=int(fpn),
        head_convs=int(spec["head"]),
        num_anchors=int(num_anchors),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 128, 128)
    m = build_yolox_detector(in_channels=3, num_classes=3, variant="yolox_tiny", width_mult=0.5)
    out = m(x)
    print("yolox_tiny", [tuple(t.shape) for t in out["obj_logits"]])
    loss = (
        sum(t.mean() for t in out["obj_logits"])
        + sum(t.mean() for t in out["cls_logits"])
        + sum(t.mean() for t in out["bbox_deltas"])
    )
    loss.backward()
    print("ok")
