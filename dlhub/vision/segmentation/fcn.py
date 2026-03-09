
import torch
from torch import nn
import torch.nn.functional as F

from dlhub.vision.backbones._blocks import scale_channels
from dlhub.vision.segmentation._common import BackboneC2C3C4C5, check_nchw


class FCNSegmenter(nn.Module):
    """FCN (Fully Convolutional Network) semantic segmentation (toy-first).

    Variants:
    - fcn32s: use C5 only
    - fcn16s: fuse C5 + C4
    - fcn8s: fuse C5 + C4 + C3
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
        mode: str = "fcn16s",
    ) -> None:
        super().__init__()
        nc = int(num_classes)
        if nc <= 0:
            raise ValueError("num_classes must be > 0")

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

        m = str(mode).lower().strip()
        if m not in {"fcn32s", "fcn16s", "fcn8s"}:
            raise ValueError("mode must be one of: fcn32s | fcn16s | fcn8s")
        self.mode = m

        self.score5 = nn.Conv2d(int(c5_channels), nc, kernel_size=1, bias=True)
        self.score4 = nn.Conv2d(int(c4_channels), nc, kernel_size=1, bias=True) if m in {"fcn16s", "fcn8s"} else None
        self.score3 = nn.Conv2d(int(c3_channels), nc, kernel_size=1, bias=True) if m in {"fcn8s"} else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = check_nchw(x)
        inp_hw = x.shape[-2:]
        _, c3, c4, c5 = self.backbone(x)

        s = self.score5(c5)
        if self.mode in {"fcn16s", "fcn8s"}:
            s = F.interpolate(s, size=c4.shape[-2:], mode="nearest") + self.score4(c4)  # type: ignore[operator]
        if self.mode in {"fcn8s"}:
            s = F.interpolate(s, size=c3.shape[-2:], mode="nearest") + self.score3(c3)  # type: ignore[operator]
        return F.interpolate(s, size=inp_hw, mode="nearest")


_VARIANTS: dict[str, dict] = {
    "fcn32s_tiny": {"stem": 24, "c2": 24, "c3": 48, "c4": 64, "c5": 96, "depth": 1, "mode": "fcn32s"},
    "fcn16s_tiny": {"stem": 24, "c2": 24, "c3": 48, "c4": 64, "c5": 96, "depth": 1, "mode": "fcn16s"},
    "fcn8s_tiny": {"stem": 24, "c2": 24, "c3": 48, "c4": 64, "c5": 96, "depth": 1, "mode": "fcn8s"},
    "fcn16s_base": {"stem": 32, "c2": 32, "c3": 64, "c4": 96, "c5": 128, "depth": 2, "mode": "fcn16s"},
}


def build_fcn_segmenter(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "fcn16s_base",
    width_mult: float = 1.0,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown FCN variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]

    stem = scale_channels(int(spec["stem"]), float(width_mult), min_ch=16, divisor=8)
    c2 = scale_channels(int(spec["c2"]), float(width_mult), min_ch=16, divisor=8)
    c3 = scale_channels(int(spec["c3"]), float(width_mult), min_ch=16, divisor=8)
    c4 = scale_channels(int(spec["c4"]), float(width_mult), min_ch=16, divisor=8)
    c5 = scale_channels(int(spec["c5"]), float(width_mult), min_ch=16, divisor=8)

    return FCNSegmenter(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        stem_channels=int(stem),
        c2_channels=int(c2),
        c3_channels=int(c3),
        c4_channels=int(c4),
        c5_channels=int(c5),
        depth=int(spec["depth"]),
        mode=str(spec["mode"]),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_fcn_segmenter(in_channels=3, num_classes=4, variant="fcn8s_tiny", width_mult=0.5)
    y = m(x)
    print("fcn8s_tiny", tuple(y.shape))
    loss = y.mean()
    loss.backward()
    print("ok")

