from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from ._utils import pad_to_multiple, unpad


def _act() -> nn.Module:
    return nn.ReLU(inplace=True)


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, *, kernel_size: int = 3, act: bool = True) -> None:
        super().__init__()
        k = int(kernel_size)
        p = k // 2
        self.conv = nn.Conv2d(int(in_ch), int(out_ch), kernel_size=k, padding=p, bias=True)
        self.act = _act() if bool(act) else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x))


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, *, reduction: int = 8) -> None:
        super().__init__()
        c = int(channels)
        r = int(reduction)
        hidden = max(8, c // max(1, r))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(c, hidden, kernel_size=1, bias=True),
            _act(),
            nn.Conv2d(hidden, c, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.mlp(self.pool(x))
        return x * w


class DualAttentionUnit(nn.Module):
    """A small residual block with channel attention (toy DU/DAU)."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        c = int(channels)
        self.conv1 = ConvBlock(c, c, kernel_size=3, act=True)
        self.conv2 = ConvBlock(c, c, kernel_size=3, act=False)
        self.ca = ChannelAttention(c, reduction=8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv2(self.conv1(x))
        y = self.ca(y)
        return x + y


class SKFF(nn.Module):
    """Selective kernel feature fusion (toy, channel-wise softmax)."""

    def __init__(self, channels: int, num_branches: int, *, reduction: int = 8) -> None:
        super().__init__()
        c = int(channels)
        b = int(num_branches)
        if b <= 1:
            raise ValueError("num_branches must be > 1")
        hidden = max(8, c // max(1, int(reduction)))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Conv2d(c, hidden, 1, bias=True), _act())
        self.fcs = nn.ModuleList([nn.Conv2d(hidden, c, 1, bias=True) for _ in range(b)])

    def forward(self, feats: list[torch.Tensor]) -> torch.Tensor:
        if len(feats) != len(self.fcs):
            raise ValueError(f"Expected {len(self.fcs)} feature maps, got {len(feats)}")
        base = feats[0]
        for f in feats[1:]:
            if f.shape != base.shape:
                raise ValueError("All SKFF branches must have the same shape")

        s = torch.zeros_like(base)
        for f in feats:
            s = s + f

        z = self.fc(self.pool(s))  # (B, hidden, 1, 1)
        logits = torch.stack([fc(z) for fc in self.fcs], dim=1)  # (B, Bn, C, 1, 1)
        w = torch.softmax(logits, dim=1)
        out = torch.zeros_like(base)
        for i, f in enumerate(feats):
            out = out + f * w[:, i]
        return out


class MultiScaleResidualBlock(nn.Module):
    """Toy MIRNet multi-scale residual block: 3-scale processing + SKFF + residual."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        c = int(channels)
        self.dau1 = DualAttentionUnit(c)
        self.dau2 = DualAttentionUnit(c)
        self.dau3 = DualAttentionUnit(c)
        self.fuse = SKFF(c, 3, reduction=8)
        self.out = ConvBlock(c, c, kernel_size=3, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape

        x1 = self.dau1(x)

        x2 = F.avg_pool2d(x, kernel_size=2, stride=2, ceil_mode=False)
        x2 = self.dau2(x2)
        x2 = F.interpolate(x2, size=(h, w), mode="nearest")

        x3 = F.avg_pool2d(x, kernel_size=4, stride=4, ceil_mode=False)
        x3 = self.dau3(x3)
        x3 = F.interpolate(x3, size=(h, w), mode="nearest")

        y = self.fuse([x1, x2, x3])
        y = self.out(y)
        return x + y


class MIRNet(nn.Module):
    """MIRNet-style denoiser (toy-first, pure torch).

    Notes:
    - This is a simplified MIRNet-inspired architecture with multi-scale residual blocks and SKFF-like fusion.
    - It performs residual learning: output = input + predicted_residual.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        width: int = 32,
        depth: int = 5,
    ) -> None:
        super().__init__()
        c_in = int(in_channels)
        w0 = int(width)
        d = int(depth)
        if c_in <= 0:
            raise ValueError("in_channels must be > 0")
        if w0 < 8:
            raise ValueError("width must be >= 8")
        if d <= 0:
            raise ValueError("depth must be > 0")

        self.intro = nn.Conv2d(c_in, w0, kernel_size=3, padding=1, bias=True)
        self.body = nn.Sequential(*[MultiScaleResidualBlock(w0) for _ in range(d)])
        self.outro = nn.Conv2d(w0, c_in, kernel_size=3, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        if x.ndim != 4:
            raise ValueError(f"Expected input shape (B, C, H, W), got {tuple(x.shape)}")

        x_pad, pad_hw = pad_to_multiple(x, 4, mode="reflect")
        inp = x_pad
        h = self.body(self.intro(x_pad))
        y = inp + self.outro(h)
        return unpad(y, pad_hw)


_VARIANTS: dict[str, dict] = {
    "mirnet_tiny": {"width": 24, "depth": 3},
    "mirnet_small": {"width": 32, "depth": 5},
    "mirnet_base": {"width": 48, "depth": 7},
}


def build_mirnet_denoiser(
    *,
    in_channels: int,
    variant: str = "mirnet_small",
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown MIRNet variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    return MIRNet(in_channels=int(in_channels), width=int(spec["width"]), depth=int(spec["depth"]))


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.rand(2, 1, 64, 64)
    noisy = (x + torch.randn_like(x) * 0.1).clamp(0.0, 1.0)
    m = build_mirnet_denoiser(in_channels=1, variant="mirnet_tiny")
    y = m(noisy)
    print("mirnet_tiny", tuple(y.shape))
    loss = (y - x).pow(2).mean()
    loss.backward()
    print("ok")

