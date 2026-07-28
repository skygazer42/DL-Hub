import torch
import torch.nn.functional as F
from torch import nn

from dlhub.vision.backbones._blocks import ConvBNAct, scale_channels
from dlhub.vision.segmentation._common import check_nchw


class _NonBottleneck1D(nn.Module):
    def __init__(self, channels: int, *, dilation: int = 1, dropout: float = 0.0) -> None:
        super().__init__()
        c = int(channels)
        d = int(dilation)
        if c <= 0:
            raise ValueError("channels must be > 0")
        if d <= 0:
            raise ValueError("dilation must be > 0")

        self.conv1 = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=(3, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, c, kernel_size=(1, 3), padding=(0, 1), bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=(3, 1), padding=(d, 0), dilation=(d, 1), bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, c, kernel_size=(1, 3), padding=(0, d), dilation=(1, d), bias=False),
            nn.BatchNorm2d(c),
        )
        self.drop = nn.Dropout2d(p=float(dropout))
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.drop(y)
        return self.act(x + y)


class ERFNet(nn.Module):
    """ERFNet semantic segmentation (compact-first)."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        base_channels: int = 24,
        depth: int = 4,
        dropout: float = 0.1,
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
        for i in range(d):
            blocks.append(_NonBottleneck1D(c, dilation=1 if i < 2 else 2, dropout=dropout))
        self.blocks = nn.Sequential(*blocks)

        self.head = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
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
    "erfnet_tiny": {"base_channels": 16, "depth": 2, "dropout": 0.0},
    "erfnet_small": {"base_channels": 24, "depth": 4, "dropout": 0.1},
    "erfnet_base": {"base_channels": 32, "depth": 6, "dropout": 0.1},
}


def build_erfnet_segmenter(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "erfnet_small",
    width_mult: float = 1.0,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown ERFNet variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    base = scale_channels(int(spec["base_channels"]), float(width_mult), min_ch=16, divisor=8)
    return ERFNet(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        base_channels=int(base),
        depth=int(spec["depth"]),
        dropout=float(spec["dropout"]),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_erfnet_segmenter(in_channels=3, num_classes=4, variant="erfnet_tiny", width_mult=0.5)
    y = m(x)
    print("erfnet_tiny", tuple(y.shape))
    loss = y.mean()
    loss.backward()
    print("ok")
