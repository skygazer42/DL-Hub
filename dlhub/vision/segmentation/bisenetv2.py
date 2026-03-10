import torch
import torch.nn.functional as F
from torch import nn

from dlhub.vision.backbones._blocks import ConvBNAct, scale_channels
from dlhub.vision.segmentation._common import check_nchw


class _SE(nn.Module):
    def __init__(self, channels: int, *, reduction: int = 8) -> None:
        super().__init__()
        c = int(channels)
        mid = max(4, c // int(reduction))
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(c, mid, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, c, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.net(x)


class _DWBlock(nn.Module):
    def __init__(self, channels: int, *, stride: int = 1) -> None:
        super().__init__()
        c = int(channels)
        s = int(stride)
        self.dw = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=3, stride=s, padding=1, groups=c, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
        )
        self.pw = ConvBNAct(c, c, kernel_size=1, stride=1, padding=0, act="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pw(self.dw(x))


class BiSeNetV2(nn.Module):
    """BiSeNetV2 semantic segmentation (toy-first).

    Detail branch (higher-res) + semantic branch (lower-res) with a lightweight fusion.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        detail_channels: int = 32,
        semantic_channels: int = 64,
        depth: int = 2,
    ) -> None:
        super().__init__()
        nc = int(num_classes)
        if nc <= 0:
            raise ValueError("num_classes must be > 0")
        dc = int(detail_channels)
        sc = int(semantic_channels)
        d = int(depth)
        if dc < 8 or sc < 8:
            raise ValueError("channels must be >= 8")
        if d <= 0:
            raise ValueError("depth must be > 0")

        # Detail branch to /4.
        self.detail = nn.Sequential(
            ConvBNAct(int(in_channels), dc, kernel_size=3, stride=2, act="relu"),  # /2
            ConvBNAct(dc, dc, kernel_size=3, stride=2, act="relu"),  # /4
            ConvBNAct(dc, dc, kernel_size=3, stride=1, act="relu"),
        )

        # Semantic branch to /8 with lightweight blocks.
        self.semantic_stem = nn.Sequential(
            ConvBNAct(int(in_channels), sc, kernel_size=3, stride=2, act="relu"),  # /2
            _DWBlock(sc, stride=2),  # /4
            _DWBlock(sc, stride=2),  # /8
        )
        self.semantic_blocks = nn.Sequential(*[_DWBlock(sc, stride=1) for _ in range(d)])

        self.fuse = nn.Sequential(
            ConvBNAct(dc + sc, sc, kernel_size=3, stride=1, act="relu"),
            _SE(sc),
            nn.Conv2d(sc, nc, kernel_size=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = check_nchw(x)
        inp_hw = x.shape[-2:]

        d = self.detail(x)  # /4
        s = self.semantic_stem(x)  # /8
        s = self.semantic_blocks(s)
        s = F.interpolate(s, size=d.shape[-2:], mode="nearest")

        logits4 = self.fuse(torch.cat([d, s], dim=1))
        return F.interpolate(logits4, size=inp_hw, mode="nearest")


_VARIANTS: dict[str, dict] = {
    "bisenetv2_tiny": {"detail": 24, "semantic": 40, "depth": 1},
    "bisenetv2_small": {"detail": 32, "semantic": 64, "depth": 2},
    "bisenetv2_base": {"detail": 48, "semantic": 80, "depth": 3},
}


def build_bisenetv2_segmenter(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "bisenetv2_small",
    width_mult: float = 1.0,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown BiSeNetV2 variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    detail = scale_channels(int(spec["detail"]), float(width_mult), min_ch=16, divisor=8)
    semantic = scale_channels(int(spec["semantic"]), float(width_mult), min_ch=16, divisor=8)
    return BiSeNetV2(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        detail_channels=int(detail),
        semantic_channels=int(semantic),
        depth=int(spec["depth"]),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_bisenetv2_segmenter(
        in_channels=3, num_classes=4, variant="bisenetv2_tiny", width_mult=0.5
    )
    y = m(x)
    print("bisenetv2_tiny", tuple(y.shape))
    loss = y.mean()
    loss.backward()
    print("ok")
