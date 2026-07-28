import torch
import torch.nn.functional as F
from torch import nn

from dlhub.vision.backbones._blocks import ConvBNAct, scale_channels
from dlhub.vision.segmentation._common import BackboneC2C3C4C5, check_nchw


class RefineNet(nn.Module):
    """RefineNet-style semantic segmentation (compact-first).

    Uses a top-down refinement pathway over multi-scale backbone features.
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
        refine_channels: int = 96,
        head_convs: int = 2,
    ) -> None:
        super().__init__()
        nc = int(num_classes)
        if nc <= 0:
            raise ValueError("num_classes must be > 0")
        rc = int(refine_channels)
        if rc <= 0:
            raise ValueError("refine_channels must be > 0")

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

        self.p2 = nn.Conv2d(int(c2_channels), rc, kernel_size=1)
        self.p3 = nn.Conv2d(int(c3_channels), rc, kernel_size=1)
        self.p4 = nn.Conv2d(int(c4_channels), rc, kernel_size=1)
        self.p5 = nn.Conv2d(int(c5_channels), rc, kernel_size=1)

        layers: list[nn.Module] = []
        for _ in range(int(head_convs)):
            layers.append(ConvBNAct(rc, rc, kernel_size=3, stride=1, act="relu"))
        layers.append(nn.Conv2d(rc, nc, kernel_size=1, bias=True))
        self.head = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = check_nchw(x)
        inp_hw = x.shape[-2:]
        c2, c3, c4, c5 = self.backbone(x)

        p5 = self.p5(c5)
        p4 = self.p4(c4) + F.interpolate(p5, size=c4.shape[-2:], mode="nearest")
        p3 = self.p3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="nearest")
        p2 = self.p2(c2) + F.interpolate(p3, size=c2.shape[-2:], mode="nearest")

        logits = self.head(p2)
        return F.interpolate(logits, size=inp_hw, mode="nearest")


_VARIANTS: dict[str, dict] = {
    "refinenet_tiny": {
        "stem": 24,
        "c2": 24,
        "c3": 48,
        "c4": 64,
        "c5": 96,
        "depth": 1,
        "refine": 64,
        "head_convs": 1,
    },
    "refinenet_small": {
        "stem": 24,
        "c2": 32,
        "c3": 64,
        "c4": 96,
        "c5": 128,
        "depth": 2,
        "refine": 96,
        "head_convs": 2,
    },
    "refinenet_base": {
        "stem": 32,
        "c2": 40,
        "c3": 80,
        "c4": 128,
        "c5": 160,
        "depth": 2,
        "refine": 128,
        "head_convs": 2,
    },
}


def build_refinenet_segmenter(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "refinenet_small",
    width_mult: float = 1.0,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown RefineNet variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]

    stem = scale_channels(int(spec["stem"]), float(width_mult), min_ch=16, divisor=8)
    c2 = scale_channels(int(spec["c2"]), float(width_mult), min_ch=16, divisor=8)
    c3 = scale_channels(int(spec["c3"]), float(width_mult), min_ch=16, divisor=8)
    c4 = scale_channels(int(spec["c4"]), float(width_mult), min_ch=16, divisor=8)
    c5 = scale_channels(int(spec["c5"]), float(width_mult), min_ch=16, divisor=8)
    refine = scale_channels(int(spec["refine"]), float(width_mult), min_ch=32, divisor=8)

    return RefineNet(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        stem_channels=int(stem),
        c2_channels=int(c2),
        c3_channels=int(c3),
        c4_channels=int(c4),
        c5_channels=int(c5),
        depth=int(spec["depth"]),
        refine_channels=int(refine),
        head_convs=int(spec["head_convs"]),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_refinenet_segmenter(
        in_channels=3, num_classes=4, variant="refinenet_tiny", width_mult=0.5
    )
    y = m(x)
    print("refinenet_tiny", tuple(y.shape))
    loss = y.mean()
    loss.backward()
    print("ok")
