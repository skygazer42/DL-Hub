import torch
import torch.nn.functional as F
from torch import nn

from dlhub.vision.backbones._blocks import ConvBNAct, scale_channels
from dlhub.vision.segmentation._common import check_nchw


class _DWSeparable(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, *, stride: int = 1) -> None:
        super().__init__()
        c_in = int(in_ch)
        c_out = int(out_ch)
        s = int(stride)
        self.dw = nn.Sequential(
            nn.Conv2d(c_in, c_in, kernel_size=3, stride=s, padding=1, groups=c_in, bias=False),
            nn.BatchNorm2d(c_in),
            nn.ReLU(inplace=True),
        )
        self.pw = ConvBNAct(c_in, c_out, kernel_size=1, stride=1, padding=0, act="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pw(self.dw(x))


class FastSCNN(nn.Module):
    """Fast-SCNN semantic segmentation (compact-first)."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        base_channels: int = 32,
        depth: int = 3,
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

        # Learning to downsample: /2 -> /4 (save) -> /8
        self.ds1 = ConvBNAct(int(in_channels), base, kernel_size=3, stride=2, act="relu")
        self.ds2 = ConvBNAct(base, base, kernel_size=3, stride=2, act="relu")
        self.ds3 = _DWSeparable(base, base * 2, stride=2)

        # Global feature extractor at /8
        gch = base * 2
        blocks: list[nn.Module] = []
        for _ in range(d):
            blocks.append(_DWSeparable(gch, gch, stride=1))
        self.global_extractor = nn.Sequential(*blocks)

        # Feature fusion: upsample global to /4 and fuse with early feature
        self.early_proj = ConvBNAct(base, base, kernel_size=1, stride=1, padding=0, act="relu")
        self.global_proj = ConvBNAct(gch, base, kernel_size=1, stride=1, padding=0, act="relu")
        self.fuse = nn.Sequential(
            ConvBNAct(base * 2, base, kernel_size=3, stride=1, act="relu"),
            nn.Conv2d(base, nc, kernel_size=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = check_nchw(x)
        inp_hw = x.shape[-2:]

        x = self.ds1(x)  # /2
        early = self.ds2(x)  # /4
        x = self.ds3(early)  # /8
        x = self.global_extractor(x)

        g = self.global_proj(x)
        g = F.interpolate(g, size=early.shape[-2:], mode="nearest")
        e = self.early_proj(early)
        logits4 = self.fuse(torch.cat([e, g], dim=1))
        return F.interpolate(logits4, size=inp_hw, mode="nearest")


_VARIANTS: dict[str, dict] = {
    "fastscnn_tiny": {"base_channels": 16, "depth": 1},
    "fastscnn_small": {"base_channels": 24, "depth": 2},
    "fastscnn_base": {"base_channels": 32, "depth": 3},
}


def build_fastscnn_segmenter(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "fastscnn_small",
    width_mult: float = 1.0,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown Fast-SCNN variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    base = scale_channels(int(spec["base_channels"]), float(width_mult), min_ch=16, divisor=8)
    return FastSCNN(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        base_channels=int(base),
        depth=int(spec["depth"]),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_fastscnn_segmenter(
        in_channels=3, num_classes=4, variant="fastscnn_tiny", width_mult=0.5
    )
    y = m(x)
    print("fastscnn_tiny", tuple(y.shape))
    loss = y.mean()
    loss.backward()
    print("ok")
