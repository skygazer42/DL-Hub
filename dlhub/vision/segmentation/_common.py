
import torch
from torch import nn
import torch.nn.functional as F

from dlhub.vision.backbones._blocks import ConvBNAct


def check_nchw(x: torch.Tensor) -> torch.Tensor:
    x = x.to(torch.float32)
    if x.ndim != 4:
        raise ValueError(f"Expected input shape (B, C, H, W), got {tuple(x.shape)}")
    return x


class ConvTower(nn.Module):
    """A small stack of Conv-BN-Act blocks."""

    def __init__(self, in_channels: int, out_channels: int, *, num_convs: int = 2, act: str = "relu") -> None:
        super().__init__()
        c_in = int(in_channels)
        c_out = int(out_channels)
        n = int(num_convs)
        if c_in <= 0 or c_out <= 0:
            raise ValueError("channels must be > 0")
        if n <= 0:
            raise ValueError("num_convs must be > 0")

        layers: list[nn.Module] = []
        ch = c_in
        for _ in range(n):
            layers.append(ConvBNAct(ch, c_out, kernel_size=3, stride=1, act=act))
            ch = c_out
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BackboneC2C3C4C5(nn.Module):
    """Tiny conv backbone that returns (C2, C3, C4, C5) feature maps.

    Strides are approximately (/4, /8, /16, /32) relative to input.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        stem_channels: int,
        c2_channels: int,
        c3_channels: int,
        c4_channels: int,
        c5_channels: int,
        depth: int,
        act: str = "relu",
    ) -> None:
        super().__init__()
        c_in = int(in_channels)
        stem = int(stem_channels)
        c2 = int(c2_channels)
        c3 = int(c3_channels)
        c4 = int(c4_channels)
        c5 = int(c5_channels)
        d = int(depth)
        if d <= 0:
            raise ValueError("depth must be > 0")

        self.stem = nn.Sequential(
            ConvBNAct(c_in, stem, kernel_size=3, stride=2, act=act),  # /2
            ConvBNAct(stem, c2, kernel_size=3, stride=2, act=act),  # /4
        )

        def stage(in_ch: int, out_ch: int) -> nn.Sequential:
            layers: list[nn.Module] = [ConvBNAct(in_ch, out_ch, kernel_size=3, stride=2, act=act)]
            for _ in range(d - 1):
                layers.append(ConvBNAct(out_ch, out_ch, kernel_size=3, stride=1, act=act))
            return nn.Sequential(*layers)

        self.stage3 = stage(c2, c3)  # /8
        self.stage4 = stage(c3, c4)  # /16
        self.stage5 = stage(c4, c5)  # /32

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        c2 = self.stem(x)
        c3 = self.stage3(c2)
        c4 = self.stage4(c3)
        c5 = self.stage5(c4)
        return c2, c3, c4, c5


class FPN4(nn.Module):
    """Minimal 4-level FPN: (C2..C5) -> (P2..P5)."""

    def __init__(self, in_channels: tuple[int, int, int, int], out_channels: int, *, act: str = "relu") -> None:
        super().__init__()
        c2, c3, c4, c5 = (int(x) for x in in_channels)
        out = int(out_channels)

        self.l2 = nn.Conv2d(c2, out, kernel_size=1)
        self.l3 = nn.Conv2d(c3, out, kernel_size=1)
        self.l4 = nn.Conv2d(c4, out, kernel_size=1)
        self.l5 = nn.Conv2d(c5, out, kernel_size=1)

        self.p2 = ConvBNAct(out, out, kernel_size=3, stride=1, act=act)
        self.p3 = ConvBNAct(out, out, kernel_size=3, stride=1, act=act)
        self.p4 = ConvBNAct(out, out, kernel_size=3, stride=1, act=act)
        self.p5 = ConvBNAct(out, out, kernel_size=3, stride=1, act=act)

    def forward(
        self, c2: torch.Tensor, c3: torch.Tensor, c4: torch.Tensor, c5: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        p5 = self.l5(c5)
        p4 = self.l4(c4) + F.interpolate(p5, size=c4.shape[-2:], mode="nearest")
        p3 = self.l3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="nearest")
        p2 = self.l2(c2) + F.interpolate(p3, size=c2.shape[-2:], mode="nearest")
        return self.p2(p2), self.p3(p3), self.p4(p4), self.p5(p5)

