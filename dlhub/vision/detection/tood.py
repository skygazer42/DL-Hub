import torch
from torch import nn

from dlhub.vision.backbones._blocks import scale_channels
from dlhub.vision.detection._common import FPN, BackboneC3C5, ConvTower, check_nchw


class TOODHead(nn.Module):
    """TOOD-style task aligned head (toy)."""

    def __init__(self, *, channels: int, num_classes: int, num_convs: int = 4) -> None:
        super().__init__()
        c = int(channels)
        nc = int(num_classes)
        n = int(num_convs)
        if nc <= 0:
            raise ValueError("num_classes must be > 0")
        self.shared = ConvTower(c, num_convs=max(1, n // 2), act="relu")
        self.cls_tower = ConvTower(c, num_convs=max(1, n // 2), act="relu")
        self.reg_tower = ConvTower(c, num_convs=max(1, n // 2), act="relu")
        self.align = nn.Conv2d(c, 1, kernel_size=3, padding=1)
        self.cls = nn.Conv2d(c, nc, kernel_size=3, padding=1)
        self.box = nn.Conv2d(c, 4, kernel_size=3, padding=1)

    def forward_single(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.shared(x)
        cls_feat = self.cls_tower(x)
        reg_feat = self.reg_tower(x)
        return {
            "align_logits": self.align(x),
            "cls_logits": self.cls(cls_feat),
            "bbox_deltas": self.box(reg_feat),
        }


class TOODDetector(nn.Module):
    """TOOD-style detector (toy-first)."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        stem_channels: int = 32,
        backbone_channels: tuple[int, int, int] = (64, 96, 128),
        backbone_depth: int = 2,
        fpn_channels: int = 96,
        head_convs: int = 4,
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
            act="relu",
        )
        self.fpn = FPN((c3, c4, c5), out, act="relu")
        self.head = TOODHead(channels=out, num_classes=int(num_classes), num_convs=int(head_convs))

    def forward(self, x: torch.Tensor) -> dict[str, list[torch.Tensor]]:
        x = check_nchw(x)
        c3, c4, c5 = self.backbone(x)
        p3, p4, p5 = self.fpn(c3, c4, c5)
        cls_out, box_out, align_out = [], [], []
        for p in (p3, p4, p5):
            y = self.head.forward_single(p)
            align_out.append(y["align_logits"])
            cls_out.append(y["cls_logits"])
            box_out.append(y["bbox_deltas"])
        return {"align_logits": align_out, "cls_logits": cls_out, "bbox_deltas": box_out}


_VARIANTS: dict[str, dict] = {
    "tood_tiny": {"stem": 24, "c3": 48, "c4": 64, "c5": 80, "depth": 1, "fpn": 64, "head": 2},
    "tood_small": {"stem": 32, "c3": 64, "c4": 96, "c5": 128, "depth": 2, "fpn": 96, "head": 4},
    "tood_base": {"stem": 48, "c3": 96, "c4": 144, "c5": 192, "depth": 3, "fpn": 128, "head": 4},
}


def build_tood_detector(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "tood_tiny",
    width_mult: float = 1.0,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown TOOD variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    stem = scale_channels(int(spec["stem"]), float(width_mult), min_ch=16, divisor=8)
    c3 = scale_channels(int(spec["c3"]), float(width_mult), min_ch=16, divisor=8)
    c4 = scale_channels(int(spec["c4"]), float(width_mult), min_ch=16, divisor=8)
    c5 = scale_channels(int(spec["c5"]), float(width_mult), min_ch=16, divisor=8)
    fpn = scale_channels(int(spec["fpn"]), float(width_mult), min_ch=16, divisor=8)
    return TOODDetector(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        stem_channels=int(stem),
        backbone_channels=(int(c3), int(c4), int(c5)),
        backbone_depth=int(spec["depth"]),
        fpn_channels=int(fpn),
        head_convs=int(spec["head"]),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 128, 128)
    m = build_tood_detector(in_channels=3, num_classes=3, variant="tood_tiny", width_mult=0.5)
    out = m(x)
    print("tood_tiny", [tuple(t.shape) for t in out["cls_logits"]])
    loss = (
        sum(t.mean() for t in out["align_logits"])
        + sum(t.mean() for t in out["cls_logits"])
        + sum(t.mean() for t in out["bbox_deltas"])
    )
    loss.backward()
    print("ok")
