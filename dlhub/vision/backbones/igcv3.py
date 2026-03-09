
import torch
from torch import nn

from dlhub.vision.backbones._blocks import ChannelShuffle, ConvBNAct, GlobalAvgPoolHead, scale_channels


class IGCV3Block(nn.Module):
    """Interleaved Group Convolutions v3 style block (simplified)."""

    def __init__(self, in_ch: int, out_ch: int, *, stride: int, groups: int = 2) -> None:
        super().__init__()
        g = int(groups)
        c_in = int(in_ch)
        c_out = int(out_ch)
        self.pw_g = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=1, groups=g, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            ChannelShuffle(g),
        )
        self.dw = nn.Sequential(
            nn.Conv2d(c_out, c_out, kernel_size=3, stride=int(stride), padding=1, groups=c_out, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )
        self.pw_g2 = nn.Sequential(
            nn.Conv2d(c_out, c_out, kernel_size=1, groups=g, bias=False),
            nn.BatchNorm2d(c_out),
        )
        self.act = nn.ReLU(inplace=True)
        self.down: nn.Module | None = None
        if int(stride) != 1 or c_in != c_out:
            self.down = nn.Sequential(nn.Conv2d(c_in, c_out, kernel_size=1, stride=int(stride), bias=False), nn.BatchNorm2d(c_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.down is None else self.down(x)
        x = self.pw_g(x)
        x = self.dw(x)
        x = self.pw_g2(x)
        return self.act(x + identity)


class IGCV3Classifier(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        channels: tuple[int, int, int, int] = (32, 64, 128, 256),
        depths: tuple[int, int, int, int] = (1, 2, 3, 2),
        groups: int = 2,
        width_mult: float = 1.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        chs = tuple(scale_channels(int(c), float(width_mult), min_ch=16, divisor=8) for c in channels)
        self.stem = ConvBNAct(int(in_channels), chs[0], kernel_size=3, stride=2, act="relu")

        def stage(in_ch: int, out_ch: int, depth: int, *, stride: int) -> nn.Sequential:
            blocks: list[nn.Module] = [IGCV3Block(in_ch, out_ch, stride=int(stride), groups=int(groups))]
            for _ in range(int(depth) - 1):
                blocks.append(IGCV3Block(out_ch, out_ch, stride=1, groups=int(groups)))
            return nn.Sequential(*blocks)

        self.stage1 = stage(chs[0], chs[0], depths[0], stride=1)
        self.stage2 = stage(chs[0], chs[1], depths[1], stride=2)
        self.stage3 = stage(chs[1], chs[2], depths[2], stride=2)
        self.stage4 = stage(chs[2], chs[3], depths[3], stride=2)
        self.head = GlobalAvgPoolHead(chs[3], int(num_classes), dropout=float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return self.head(x)


_VARIANTS: dict[str, dict] = {
    "igcv3_tiny": {"channels": (24, 48, 96, 192), "depths": (1, 2, 3, 2), "groups": 2},
    "igcv3_base": {"channels": (32, 64, 128, 256), "depths": (1, 2, 3, 2), "groups": 2},
    "igcv3_wide": {"channels": (48, 96, 192, 384), "depths": (2, 2, 4, 2), "groups": 4},
}


def build_igcv3_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "igcv3_base",
    width_mult: float = 1.0,
    dropout: float = 0.1,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown IGCV3 variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    return IGCV3Classifier(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        channels=tuple(map(int, spec["channels"])),
        depths=tuple(map(int, spec["depths"])),
        groups=int(spec["groups"]),
        width_mult=float(width_mult),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_igcv3_classifier(in_channels=3, num_classes=10, variant="igcv3_base", width_mult=0.5)
    y = m(x)
    print("igcv3_base", tuple(y.shape))

