
import torch
from torch import nn
import torch.nn.functional as F

from dlhub.vision.backbones._blocks import ConvBNAct, scale_channels
from dlhub.vision.segmentation._common import check_nchw


class _AsymBlock(nn.Module):
    def __init__(self, channels: int, *, dilation: int = 1) -> None:
        super().__init__()
        c = int(channels)
        d = int(dilation)
        self.net = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=(3, 1), padding=(d, 0), dilation=(d, 1), bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, c, kernel_size=(1, 3), padding=(0, d), dilation=(1, d), bias=False),
            nn.BatchNorm2d(c),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class LEDNet(nn.Module):
    """LEDNet semantic segmentation (toy-first)."""

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

        self.enc = nn.Sequential(
            ConvBNAct(int(in_channels), base, kernel_size=3, stride=2, act="relu"),  # /2
            ConvBNAct(base, base * 2, kernel_size=3, stride=2, act="relu"),  # /4
            ConvBNAct(base * 2, base * 4, kernel_size=3, stride=2, act="relu"),  # /8
        )
        c = base * 4
        blocks: list[nn.Module] = []
        for i in range(d):
            blocks.append(_AsymBlock(c, dilation=1 if i < 2 else 2))
        self.blocks = nn.Sequential(*blocks)

        self.dec = nn.Sequential(
            ConvBNAct(c, base * 2, kernel_size=3, stride=1, act="relu"),
            ConvBNAct(base * 2, base, kernel_size=3, stride=1, act="relu"),
            nn.Conv2d(base, nc, kernel_size=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = check_nchw(x)
        inp_hw = x.shape[-2:]
        x = self.enc(x)
        x = self.blocks(x)
        x = F.interpolate(x, scale_factor=2, mode="nearest")  # /4
        logits4 = self.dec(x)
        return F.interpolate(logits4, size=inp_hw, mode="nearest")


_VARIANTS: dict[str, dict] = {
    "lednet_tiny": {"base_channels": 16, "depth": 2},
    "lednet_small": {"base_channels": 24, "depth": 4},
    "lednet_base": {"base_channels": 32, "depth": 6},
}


def build_lednet_segmenter(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "lednet_small",
    width_mult: float = 1.0,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown LEDNet variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    base = scale_channels(int(spec["base_channels"]), float(width_mult), min_ch=16, divisor=8)
    return LEDNet(in_channels=int(in_channels), num_classes=int(num_classes), base_channels=int(base), depth=int(spec["depth"]))


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_lednet_segmenter(in_channels=3, num_classes=4, variant="lednet_tiny", width_mult=0.5)
    y = m(x)
    print("lednet_tiny", tuple(y.shape))
    loss = y.mean()
    loss.backward()
    print("ok")

