import torch
import torch.nn.functional as F
from torch import nn

from dlhub.vision.backbones._blocks import ConvBNAct, scale_channels
from dlhub.vision.segmentation._common import check_nchw


class _ENetBottleneck(nn.Module):
    def __init__(
        self, channels: int, *, bottleneck: int, dilation: int = 1, dropout: float = 0.0
    ) -> None:
        super().__init__()
        c = int(channels)
        b = int(bottleneck)
        dil = int(dilation)
        if c <= 0 or b <= 0:
            raise ValueError("channels must be > 0")
        if dil <= 0:
            raise ValueError("dilation must be > 0")

        self.reduce = ConvBNAct(c, b, kernel_size=1, stride=1, act="relu")
        self.conv = nn.Sequential(
            nn.Conv2d(b, b, kernel_size=3, padding=dil, dilation=dil, bias=False),
            nn.BatchNorm2d(b),
            nn.ReLU(inplace=True),
        )
        self.expand = nn.Sequential(
            nn.Conv2d(b, c, kernel_size=1, bias=False),
            nn.BatchNorm2d(c),
        )
        self.drop = nn.Dropout2d(p=float(dropout))
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.reduce(x)
        y = self.conv(y)
        y = self.expand(y)
        y = self.drop(y)
        return self.act(x + y)


class _Downsample(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        c_in = int(in_ch)
        c_out = int(out_ch)
        self.conv = ConvBNAct(c_in, c_out, kernel_size=3, stride=2, act="relu")
        self.skip = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(c_in, c_out, kernel_size=1, bias=False),
            nn.BatchNorm2d(c_out),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x) + self.skip(x))


class ENet(nn.Module):
    """ENet semantic segmentation (compact-first, pure torch).

    Compact, downsampling-heavy network; predicts logits and upsamples to input.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        base_channels: int = 24,
        stage_depth: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        nc = int(num_classes)
        if nc <= 0:
            raise ValueError("num_classes must be > 0")
        base = int(base_channels)
        d = int(stage_depth)
        if base < 8:
            raise ValueError("base_channels must be >= 8")
        if d <= 0:
            raise ValueError("stage_depth must be > 0")

        self.stem = nn.Sequential(
            ConvBNAct(int(in_channels), base, kernel_size=3, stride=2, act="relu"),  # /2
            ConvBNAct(base, base * 2, kernel_size=3, stride=2, act="relu"),  # /4
        )

        c = base * 2
        self.down = _Downsample(c, c * 2)  # /8
        c *= 2

        blocks: list[nn.Module] = []
        for i in range(d):
            blocks.append(
                _ENetBottleneck(
                    c, bottleneck=max(8, c // 4), dilation=1 if i % 2 == 0 else 2, dropout=dropout
                )
            )
        self.blocks = nn.Sequential(*blocks)

        self.head = nn.Sequential(
            ConvBNAct(c, c, kernel_size=3, stride=1, act="relu"),
            nn.Conv2d(c, nc, kernel_size=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = check_nchw(x)
        inp_hw = x.shape[-2:]
        x = self.stem(x)
        x = self.down(x)
        x = self.blocks(x)
        logits = self.head(x)
        return F.interpolate(logits, size=inp_hw, mode="nearest")


_VARIANTS: dict[str, dict] = {
    "enet_tiny": {"base_channels": 16, "stage_depth": 2, "dropout": 0.0},
    "enet_small": {"base_channels": 24, "stage_depth": 3, "dropout": 0.1},
    "enet_base": {"base_channels": 32, "stage_depth": 4, "dropout": 0.1},
}


def build_enet_segmenter(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "enet_small",
    width_mult: float = 1.0,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown ENet variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    base = scale_channels(int(spec["base_channels"]), float(width_mult), min_ch=16, divisor=8)
    return ENet(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        base_channels=int(base),
        stage_depth=int(spec["stage_depth"]),
        dropout=float(spec["dropout"]),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_enet_segmenter(in_channels=3, num_classes=4, variant="enet_tiny", width_mult=0.5)
    y = m(x)
    print("enet_tiny", tuple(y.shape))
    loss = y.mean()
    loss.backward()
    print("ok")
