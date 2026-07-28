import torch
import torch.nn.functional as F
from torch import nn


class DirectionalMaskedConv2d(nn.Module):
    """A conv2d whose kernel is masked to enforce a directional receptive field.

    Directions:
    - "up":    uses only rows strictly above the center row
    - "down":  uses only rows strictly below the center row
    - "left":  uses only cols strictly left of the center col
    - "right": uses only cols strictly right of the center col

    This is a compact-first building block for blind-spot networks. The mask is applied
    in forward (weight * mask) so masked weights never contribute.
    """

    def __init__(self, in_ch: int, out_ch: int, *, kernel_size: int = 3, direction: str) -> None:
        super().__init__()
        k = int(kernel_size)
        if k < 3 or k % 2 == 0:
            raise ValueError("kernel_size must be odd and >= 3")
        direction = str(direction).lower().strip()
        if direction not in {"up", "down", "left", "right"}:
            raise ValueError(f"Unknown direction: {direction!r}")

        p = k // 2
        self.conv = nn.Conv2d(int(in_ch), int(out_ch), kernel_size=k, padding=p, bias=True)
        mask = torch.ones_like(self.conv.weight)

        center = p
        if direction == "up":
            # Only rows < center.
            mask[:, :, center:, :] = 0
        elif direction == "down":
            # Only rows > center.
            mask[:, :, : center + 1, :] = 0
        elif direction == "left":
            # Only cols < center.
            mask[:, :, :, center:] = 0
        elif direction == "right":
            # Only cols > center.
            mask[:, :, :, : center + 1] = 0

        self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.conv.weight * self.mask
        return F.conv2d(x, w, bias=self.conv.bias, stride=1, padding=self.conv.padding)


class DirectionalResBlock(nn.Module):
    def __init__(self, channels: int, *, direction: str) -> None:
        super().__init__()
        c = int(channels)
        self.conv1 = DirectionalMaskedConv2d(c, c, kernel_size=3, direction=str(direction))
        self.conv2 = DirectionalMaskedConv2d(c, c, kernel_size=3, direction=str(direction))
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act(self.conv1(x))
        y = self.conv2(y)
        return self.act(x + y)


class DirectionalStack(nn.Module):
    def __init__(self, in_channels: int, width: int, *, depth: int, direction: str) -> None:
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

        self.in_conv = DirectionalMaskedConv2d(c_in, w0, kernel_size=3, direction=str(direction))
        self.blocks = nn.Sequential(
            *[DirectionalResBlock(w0, direction=str(direction)) for _ in range(d)]
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.in_conv(x))
        return self.blocks(x)


class BlindSpotNet(nn.Module):
    """Blind-Spot Network (BSN) style denoiser (compact-first, pure torch).

    This is a simplified directional blind-spot network:
    - Four directional stacks (up/down/left/right) with masked convs that exclude the center pixel.
    - Fuse directional features and predict a residual/noise map.
    - Output is `x - residual`.

    It is designed to pair well with blind-spot self-supervised training, but can also be trained
    supervised or noise2noise on synthetic data.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        width: int = 24,
        depth: int = 4,
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

        self.up = DirectionalStack(c_in, w0, depth=d, direction="up")
        self.down = DirectionalStack(c_in, w0, depth=d, direction="down")
        self.left = DirectionalStack(c_in, w0, depth=d, direction="left")
        self.right = DirectionalStack(c_in, w0, depth=d, direction="right")

        self.fuse = nn.Sequential(
            nn.Conv2d(w0 * 4, w0, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(w0, c_in, kernel_size=3, padding=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        if x.ndim != 4:
            raise ValueError(f"Expected input shape (B, C, H, W), got {tuple(x.shape)}")

        f = torch.cat([self.up(x), self.down(x), self.left(x), self.right(x)], dim=1)
        residual = self.fuse(f)
        return x - residual


_VARIANTS: dict[str, dict] = {
    "bsn_tiny": {"width": 16, "depth": 2},
    "bsn_small": {"width": 24, "depth": 4},
    "bsn_base": {"width": 32, "depth": 6},
}


def build_bsn_denoiser(
    *,
    in_channels: int,
    variant: str = "bsn_small",
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown BSN variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    return BlindSpotNet(
        in_channels=int(in_channels), width=int(spec["width"]), depth=int(spec["depth"])
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.rand(2, 1, 64, 64)
    noisy = (x + torch.randn_like(x) * 0.1).clamp(0.0, 1.0)
    m = build_bsn_denoiser(in_channels=1, variant="bsn_tiny")
    y = m(noisy)
    print("bsn_tiny", tuple(y.shape))
    loss = (y - x).pow(2).mean()
    loss.backward()
    print("ok")
