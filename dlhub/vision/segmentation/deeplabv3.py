import torch
import torch.nn.functional as F
from torch import nn

from dlhub.vision.backbones._blocks import ConvBNAct, scale_channels
from dlhub.vision.segmentation._common import BackboneC2C3C4C5, check_nchw


class ASPP(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, *, rates: tuple[int, int, int]) -> None:
        super().__init__()
        c_in = int(in_ch)
        c_out = int(out_ch)
        r1, r2, r3 = (int(r) for r in rates)

        self.b0 = ConvBNAct(c_in, c_out, kernel_size=1, stride=1, padding=0, act="relu")
        self.d1 = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=3, padding=r1, dilation=r1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )
        self.d2 = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=3, padding=r2, dilation=r2, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )
        self.d3 = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=3, padding=r3, dilation=r3, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )

        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(c_in, c_out, kernel_size=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )

        self.project = nn.Sequential(
            nn.Conv2d(c_out * 5, c_out, kernel_size=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b0 = self.b0(x)
        d1 = self.d1(x)
        d2 = self.d2(x)
        d3 = self.d3(x)
        p = self.pool(x)
        p = F.interpolate(p, size=x.shape[-2:], mode="nearest")
        return self.project(torch.cat([b0, d1, d2, d3, p], dim=1))


class DeepLabV3(nn.Module):
    """DeepLabV3 semantic segmentation (toy-first, pure torch)."""

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
        aspp_channels: int = 128,
        aspp_rates: tuple[int, int, int] = (6, 12, 18),
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

        self.aspp = ASPP(
            int(c4_channels), int(aspp_channels), rates=tuple(int(r) for r in aspp_rates)
        )
        self.head = nn.Sequential(
            ConvBNAct(int(aspp_channels), int(aspp_channels), kernel_size=3, stride=1, act="relu"),
            nn.Conv2d(int(aspp_channels), nc, kernel_size=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = check_nchw(x)
        inp_hw = x.shape[-2:]
        _, _, c4, _ = self.backbone(x)
        y = self.aspp(c4)
        logits = self.head(y)
        return F.interpolate(logits, size=inp_hw, mode="nearest")


_VARIANTS: dict[str, dict] = {
    "deeplabv3_tiny": {
        "stem": 24,
        "c2": 24,
        "c3": 48,
        "c4": 64,
        "c5": 96,
        "depth": 1,
        "aspp": 96,
        "rates": (3, 6, 9),
    },
    "deeplabv3_small": {
        "stem": 24,
        "c2": 32,
        "c3": 64,
        "c4": 96,
        "c5": 128,
        "depth": 2,
        "aspp": 128,
        "rates": (4, 8, 12),
    },
    "deeplabv3_base": {
        "stem": 32,
        "c2": 40,
        "c3": 80,
        "c4": 128,
        "c5": 160,
        "depth": 2,
        "aspp": 160,
        "rates": (6, 12, 18),
    },
}


def build_deeplabv3_segmenter(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "deeplabv3_small",
    width_mult: float = 1.0,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown DeepLabV3 variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]

    stem = scale_channels(int(spec["stem"]), float(width_mult), min_ch=16, divisor=8)
    c2 = scale_channels(int(spec["c2"]), float(width_mult), min_ch=16, divisor=8)
    c3 = scale_channels(int(spec["c3"]), float(width_mult), min_ch=16, divisor=8)
    c4 = scale_channels(int(spec["c4"]), float(width_mult), min_ch=16, divisor=8)
    c5 = scale_channels(int(spec["c5"]), float(width_mult), min_ch=16, divisor=8)
    aspp = scale_channels(int(spec["aspp"]), float(width_mult), min_ch=32, divisor=8)

    return DeepLabV3(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        stem_channels=int(stem),
        c2_channels=int(c2),
        c3_channels=int(c3),
        c4_channels=int(c4),
        c5_channels=int(c5),
        depth=int(spec["depth"]),
        aspp_channels=int(aspp),
        aspp_rates=tuple(int(r) for r in spec["rates"]),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_deeplabv3_segmenter(
        in_channels=3, num_classes=4, variant="deeplabv3_tiny", width_mult=0.5
    )
    y = m(x)
    print("deeplabv3_tiny", tuple(y.shape))
    loss = y.mean()
    loss.backward()
    print("ok")
