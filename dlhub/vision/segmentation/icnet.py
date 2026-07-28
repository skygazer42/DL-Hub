import torch
import torch.nn.functional as F
from torch import nn

from dlhub.vision.backbones._blocks import ConvBNAct, scale_channels
from dlhub.vision.segmentation._common import check_nchw


class _Branch(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, *, depth: int) -> None:
        super().__init__()
        c_in = int(in_ch)
        c_out = int(out_ch)
        d = int(depth)
        if d <= 0:
            raise ValueError("depth must be > 0")
        layers: list[nn.Module] = [ConvBNAct(c_in, c_out, kernel_size=3, stride=1, act="relu")]
        for _ in range(d - 1):
            layers.append(ConvBNAct(c_out, c_out, kernel_size=3, stride=1, act="relu"))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ICNet(nn.Module):
    """ICNet semantic segmentation (compact-first).

    Multi-resolution branches (1x, 1/2x, 1/4x) fused coarse-to-fine.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        branch_channels: int = 48,
        depth: int = 2,
    ) -> None:
        super().__init__()
        nc = int(num_classes)
        if nc <= 0:
            raise ValueError("num_classes must be > 0")
        ch = int(branch_channels)
        d = int(depth)
        if ch < 8:
            raise ValueError("branch_channels must be >= 8")
        if d <= 0:
            raise ValueError("depth must be > 0")

        self.b1 = _Branch(int(in_channels), ch, depth=d)  # full-res
        self.b2 = _Branch(int(in_channels), ch, depth=d)  # half-res
        self.b3 = _Branch(int(in_channels), ch, depth=d)  # quarter-res

        self.fuse32 = ConvBNAct(ch, ch, kernel_size=3, stride=1, act="relu")
        self.fuse21 = ConvBNAct(ch, ch, kernel_size=3, stride=1, act="relu")
        self.out = nn.Conv2d(ch, nc, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = check_nchw(x)
        inp_hw = x.shape[-2:]

        x1 = self.b1(x)
        x2 = self.b2(F.interpolate(x, scale_factor=0.5, mode="nearest"))
        x3 = self.b3(F.interpolate(x, scale_factor=0.25, mode="nearest"))

        x3_up = F.interpolate(x3, size=x2.shape[-2:], mode="nearest")
        f2 = self.fuse32(x2 + x3_up)
        f2_up = F.interpolate(f2, size=x1.shape[-2:], mode="nearest")
        f1 = self.fuse21(x1 + f2_up)

        logits = self.out(f1)
        return F.interpolate(logits, size=inp_hw, mode="nearest")


_VARIANTS: dict[str, dict] = {
    "icnet_tiny": {"branch_channels": 32, "depth": 1},
    "icnet_small": {"branch_channels": 48, "depth": 2},
    "icnet_base": {"branch_channels": 64, "depth": 3},
}


def build_icnet_segmenter(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "icnet_small",
    width_mult: float = 1.0,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown ICNet variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    ch = scale_channels(int(spec["branch_channels"]), float(width_mult), min_ch=16, divisor=8)
    return ICNet(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        branch_channels=int(ch),
        depth=int(spec["depth"]),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_icnet_segmenter(in_channels=3, num_classes=4, variant="icnet_tiny", width_mult=0.5)
    y = m(x)
    print("icnet_tiny", tuple(y.shape))
    loss = y.mean()
    loss.backward()
    print("ok")
