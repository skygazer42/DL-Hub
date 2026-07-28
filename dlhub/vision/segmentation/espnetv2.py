import torch
import torch.nn.functional as F
from torch import nn

from dlhub.vision.backbones._blocks import ConvBNAct, scale_channels
from dlhub.vision.segmentation._common import check_nchw


class _EESP(nn.Module):
    """EESP block (simplified): pointwise -> parallel dilated depthwise -> pointwise."""

    def __init__(
        self, channels: int, *, rates: tuple[int, ...] = (1, 2, 4), expansion: float = 0.5
    ) -> None:
        super().__init__()
        c = int(channels)
        if c <= 0:
            raise ValueError("channels must be > 0")
        rs = tuple(int(r) for r in rates)
        if not rs:
            raise ValueError("rates must be non-empty")

        mid = max(8, int(round(c * float(expansion))))
        self.pw1 = ConvBNAct(c, mid, kernel_size=1, stride=1, padding=0, act="relu")
        self.dw = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        mid, mid, kernel_size=3, padding=r, dilation=r, groups=mid, bias=False
                    ),
                    nn.BatchNorm2d(mid),
                    nn.ReLU(inplace=True),
                )
                for r in rs
            ]
        )
        self.pw2 = nn.Sequential(
            nn.Conv2d(mid * len(rs), c, kernel_size=1, bias=False),
            nn.BatchNorm2d(c),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.pw1(x)
        ys = [b(y) for b in self.dw]
        y = self.pw2(torch.cat(ys, dim=1))
        return self.act(x + y)


class ESPNetV2(nn.Module):
    """ESPNetV2 semantic segmentation (compact-first)."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        base_channels: int = 24,
        depth: int = 4,
    ) -> None:
        super().__init__()
        nc = int(num_classes)
        if nc <= 0:
            raise ValueError("num_classes must be > 0")
        base = int(base_channels)
        d = int(depth)
        if base < 8:
            raise ValueError("base_channels must be >= 8")
        if d <= 0:
            raise ValueError("depth must be > 0")

        self.stem = nn.Sequential(
            ConvBNAct(int(in_channels), base, kernel_size=3, stride=2, act="relu"),  # /2
            ConvBNAct(base, base * 2, kernel_size=3, stride=2, act="relu"),  # /4
            ConvBNAct(base * 2, base * 4, kernel_size=3, stride=2, act="relu"),  # /8
        )
        c = base * 4
        blocks: list[nn.Module] = []
        for _ in range(d):
            blocks.append(_EESP(c, rates=(1, 2, 4), expansion=0.5))
        self.blocks = nn.Sequential(*blocks)

        self.head = nn.Sequential(
            ConvBNAct(c, c, kernel_size=3, stride=1, act="relu"),
            nn.Conv2d(c, nc, kernel_size=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = check_nchw(x)
        inp_hw = x.shape[-2:]
        x = self.stem(x)
        x = self.blocks(x)
        logits = self.head(x)
        return F.interpolate(logits, size=inp_hw, mode="nearest")


_VARIANTS: dict[str, dict] = {
    "espnetv2_tiny": {"base_channels": 16, "depth": 2},
    "espnetv2_small": {"base_channels": 24, "depth": 4},
    "espnetv2_base": {"base_channels": 32, "depth": 6},
}


def build_espnetv2_segmenter(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "espnetv2_small",
    width_mult: float = 1.0,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown ESPNetV2 variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    base = scale_channels(int(spec["base_channels"]), float(width_mult), min_ch=16, divisor=8)
    return ESPNetV2(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        base_channels=int(base),
        depth=int(spec["depth"]),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_espnetv2_segmenter(
        in_channels=3, num_classes=4, variant="espnetv2_tiny", width_mult=0.5
    )
    y = m(x)
    print("espnetv2_tiny", tuple(y.shape))
    loss = y.mean()
    loss.backward()
    print("ok")
