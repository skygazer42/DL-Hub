import torch
from torch import nn

from dlhub.vision.backbones._blocks import scale_channels
from dlhub.vision.detection._common import FPN, BackboneC3C5, ConvTower, check_nchw


class RepPointsHead(nn.Module):
    """RepPoints-style point set head (compact)."""

    def __init__(
        self, *, channels: int, num_classes: int, num_points: int = 9, num_convs: int = 3
    ) -> None:
        super().__init__()
        c = int(channels)
        nc = int(num_classes)
        p = int(num_points)
        n = int(num_convs)
        if nc <= 0:
            raise ValueError("num_classes must be > 0")
        if p <= 0:
            raise ValueError("num_points must be > 0")
        self.num_points = p

        self.shared = ConvTower(c, num_convs=n, act="relu")
        self.cls = nn.Conv2d(c, nc, kernel_size=3, padding=1)
        self.pts_init = nn.Conv2d(c, 2 * p, kernel_size=3, padding=1)
        self.pts_refine = nn.Conv2d(c, 2 * p, kernel_size=3, padding=1)

    def forward_single(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.shared(x)
        return {
            "cls_logits": self.cls(x),
            "points_init": self.pts_init(x),
            "points_refine": self.pts_refine(x),
        }


class RepPointsDetector(nn.Module):
    """RepPoints-style detector (compact-first)."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        stem_channels: int = 32,
        backbone_channels: tuple[int, int, int] = (64, 96, 128),
        backbone_depth: int = 2,
        fpn_channels: int = 96,
        head_convs: int = 3,
        num_points: int = 9,
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
        self.head = RepPointsHead(
            channels=out,
            num_classes=int(num_classes),
            num_points=int(num_points),
            num_convs=int(head_convs),
        )

    def forward(self, x: torch.Tensor) -> dict[str, list[torch.Tensor]]:
        x = check_nchw(x)
        c3, c4, c5 = self.backbone(x)
        p3, p4, p5 = self.fpn(c3, c4, c5)
        cls_out, init_out, ref_out = [], [], []
        for p in (p3, p4, p5):
            y = self.head.forward_single(p)
            cls_out.append(y["cls_logits"])
            init_out.append(y["points_init"])
            ref_out.append(y["points_refine"])
        return {"cls_logits": cls_out, "points_init": init_out, "points_refine": ref_out}


_VARIANTS: dict[str, dict] = {
    "reppoints_tiny": {
        "stem": 24,
        "c3": 48,
        "c4": 64,
        "c5": 80,
        "depth": 1,
        "fpn": 64,
        "head": 2,
        "points": 9,
    },
    "reppoints_small": {
        "stem": 32,
        "c3": 64,
        "c4": 96,
        "c5": 128,
        "depth": 2,
        "fpn": 96,
        "head": 3,
        "points": 9,
    },
    "reppoints_base": {
        "stem": 48,
        "c3": 96,
        "c4": 144,
        "c5": 192,
        "depth": 3,
        "fpn": 128,
        "head": 3,
        "points": 9,
    },
}


def build_reppoints_detector(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "reppoints_tiny",
    width_mult: float = 1.0,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown RepPoints variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    stem = scale_channels(int(spec["stem"]), float(width_mult), min_ch=16, divisor=8)
    c3 = scale_channels(int(spec["c3"]), float(width_mult), min_ch=16, divisor=8)
    c4 = scale_channels(int(spec["c4"]), float(width_mult), min_ch=16, divisor=8)
    c5 = scale_channels(int(spec["c5"]), float(width_mult), min_ch=16, divisor=8)
    fpn = scale_channels(int(spec["fpn"]), float(width_mult), min_ch=16, divisor=8)
    return RepPointsDetector(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        stem_channels=int(stem),
        backbone_channels=(int(c3), int(c4), int(c5)),
        backbone_depth=int(spec["depth"]),
        fpn_channels=int(fpn),
        head_convs=int(spec["head"]),
        num_points=int(spec["points"]),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 128, 128)
    m = build_reppoints_detector(
        in_channels=3, num_classes=3, variant="reppoints_tiny", width_mult=0.5
    )
    out = m(x)
    print("reppoints_tiny", [tuple(t.shape) for t in out["cls_logits"]])
    loss = (
        sum(t.mean() for t in out["cls_logits"])
        + sum(t.mean() for t in out["points_init"])
        + sum(t.mean() for t in out["points_refine"])
    )
    loss.backward()
    print("ok")
