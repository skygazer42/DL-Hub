
import torch
from torch import nn
import torch.nn.functional as F

from dlhub.vision.backbones._blocks import ConvBNAct, scale_channels
from dlhub.vision.segmentation._common import check_nchw


class _ESPBlock(nn.Module):
    """Efficient Spatial Pyramid block (simplified)."""

    def __init__(self, in_ch: int, out_ch: int, *, rates: tuple[int, ...] = (1, 2, 4, 8)) -> None:
        super().__init__()
        c_in = int(in_ch)
        c_out = int(out_ch)
        if c_in <= 0 or c_out <= 0:
            raise ValueError("channels must be > 0")
        rs = tuple(int(r) for r in rates)
        if not rs:
            raise ValueError("rates must be non-empty")

        b = max(8, c_out // len(rs))
        self.reduce = ConvBNAct(c_in, b, kernel_size=1, stride=1, padding=0, act="relu")
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(b, b, kernel_size=3, padding=r, dilation=r, bias=False),
                    nn.BatchNorm2d(b),
                    nn.ReLU(inplace=True),
                )
                for r in rs
            ]
        )
        self.project = nn.Sequential(
            nn.Conv2d(b * len(rs), c_out, kernel_size=1, bias=False),
            nn.BatchNorm2d(c_out),
        )
        self.skip = nn.Conv2d(c_in, c_out, kernel_size=1, bias=False) if c_in != c_out else nn.Identity()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.reduce(x)
        ys = [b(y) for b in self.branches]
        y = self.project(torch.cat(ys, dim=1))
        return self.act(y + self.skip(x))


class ESPNet(nn.Module):
    """ESPNet semantic segmentation (toy-first)."""

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
        )
        self.down = ConvBNAct(base * 2, base * 4, kernel_size=3, stride=2, act="relu")  # /8
        c = base * 4

        blocks: list[nn.Module] = []
        for _ in range(d):
            blocks.append(_ESPBlock(c, c, rates=(1, 2, 4, 8)))
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
    "espnet_tiny": {"base_channels": 16, "depth": 2},
    "espnet_small": {"base_channels": 24, "depth": 4},
    "espnet_base": {"base_channels": 32, "depth": 6},
}


def build_espnet_segmenter(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "espnet_small",
    width_mult: float = 1.0,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown ESPNet variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    base = scale_channels(int(spec["base_channels"]), float(width_mult), min_ch=16, divisor=8)
    return ESPNet(in_channels=int(in_channels), num_classes=int(num_classes), base_channels=int(base), depth=int(spec["depth"]))


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_espnet_segmenter(in_channels=3, num_classes=4, variant="espnet_tiny", width_mult=0.5)
    y = m(x)
    print("espnet_tiny", tuple(y.shape))
    loss = y.mean()
    loss.backward()
    print("ok")

