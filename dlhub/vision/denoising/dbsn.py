
import torch
from torch import nn
import torch.nn.functional as F


class DirectionalMaskedConv2d(nn.Module):
    """Directional masked conv for blind-spot networks (supports dilation).

    Directions:
    - "up":    uses only rows strictly above the center row
    - "down":  uses only rows strictly below the center row
    - "left":  uses only cols strictly left of the center col
    - "right": uses only cols strictly right of the center col
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        *,
        kernel_size: int = 3,
        dilation: int = 1,
        direction: str,
    ) -> None:
        super().__init__()
        k = int(kernel_size)
        d = int(dilation)
        if k < 3 or k % 2 == 0:
            raise ValueError("kernel_size must be odd and >= 3")
        if d <= 0:
            raise ValueError("dilation must be > 0")

        direction = str(direction).lower().strip()
        if direction not in {"up", "down", "left", "right"}:
            raise ValueError(f"Unknown direction: {direction!r}")

        p = (k // 2) * d
        self.conv = nn.Conv2d(int(in_ch), int(out_ch), kernel_size=k, padding=p, dilation=d, bias=True)

        mask = torch.ones_like(self.conv.weight)
        center = k // 2
        if direction == "up":
            mask[:, :, center:, :] = 0
        elif direction == "down":
            mask[:, :, : center + 1, :] = 0
        elif direction == "left":
            mask[:, :, :, center:] = 0
        else:
            mask[:, :, :, : center + 1] = 0

        self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.conv.weight * self.mask
        return F.conv2d(
            x,
            w,
            bias=self.conv.bias,
            stride=self.conv.stride,
            padding=self.conv.padding,
            dilation=self.conv.dilation,
            groups=self.conv.groups,
        )


class DirectionalResBlock(nn.Module):
    def __init__(self, channels: int, *, direction: str, dilation: int) -> None:
        super().__init__()
        c = int(channels)
        self.conv1 = DirectionalMaskedConv2d(c, c, kernel_size=3, dilation=int(dilation), direction=str(direction))
        self.conv2 = DirectionalMaskedConv2d(c, c, kernel_size=3, dilation=int(dilation), direction=str(direction))
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act(self.conv1(x))
        y = self.conv2(y)
        return self.act(x + y)


class DirectionalStack(nn.Module):
    def __init__(
        self,
        in_channels: int,
        width: int,
        *,
        depth: int,
        direction: str,
        dilation: int,
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

        self.in_conv = DirectionalMaskedConv2d(
            c_in,
            w0,
            kernel_size=3,
            dilation=int(dilation),
            direction=str(direction),
        )
        self.blocks = nn.Sequential(
            *[
                DirectionalResBlock(w0, direction=str(direction), dilation=int(dilation))
                for _ in range(d)
            ]
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.in_conv(x))
        return self.blocks(x)


class DBSN(nn.Module):
    """Dilated Blind-Spot Network (toy-first, pure torch).

    Compared to `BSN`, this version:
    - Runs multiple directional stacks at different dilation rates.
    - Fuses all directional+dilation features to predict residual/noise.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        width: int = 24,
        depth: int = 3,
        dilations: tuple[int, ...] = (1, 2),
    ) -> None:
        super().__init__()
        c_in = int(in_channels)
        w0 = int(width)
        d = int(depth)
        dil = tuple(int(x) for x in dilations)
        if c_in <= 0:
            raise ValueError("in_channels must be > 0")
        if w0 < 8:
            raise ValueError("width must be >= 8")
        if d <= 0:
            raise ValueError("depth must be > 0")
        if not dil or any(x <= 0 for x in dil):
            raise ValueError("dilations must be positive")

        stacks: list[nn.Module] = []
        for direction in ("up", "down", "left", "right"):
            for di in dil:
                stacks.append(
                    DirectionalStack(
                        c_in,
                        w0,
                        depth=d,
                        direction=direction,
                        dilation=di,
                    )
                )
        self.stacks = nn.ModuleList(stacks)

        self.fuse = nn.Sequential(
            nn.Conv2d(w0 * len(stacks), w0, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(w0, c_in, kernel_size=3, padding=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        if x.ndim != 4:
            raise ValueError(f"Expected input shape (B, C, H, W), got {tuple(x.shape)}")

        feats = [m(x) for m in self.stacks]
        f = torch.cat(feats, dim=1)
        residual = self.fuse(f)
        return x - residual


_VARIANTS: dict[str, dict] = {
    "dbsn_tiny": {"width": 16, "depth": 2, "dilations": (1, 2)},
    "dbsn_small": {"width": 24, "depth": 3, "dilations": (1, 2)},
    "dbsn_base": {"width": 32, "depth": 4, "dilations": (1, 2, 3)},
}


def build_dbsn_denoiser(
    *,
    in_channels: int,
    variant: str = "dbsn_small",
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown DBSN variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    return DBSN(
        in_channels=int(in_channels),
        width=int(spec["width"]),
        depth=int(spec["depth"]),
        dilations=tuple(map(int, spec["dilations"])),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.rand(2, 1, 64, 64)
    noisy = (x + torch.randn_like(x) * 0.1).clamp(0.0, 1.0)
    m = build_dbsn_denoiser(in_channels=1, variant="dbsn_tiny")
    y = m(noisy)
    print("dbsn_tiny", tuple(y.shape))
    loss = (y - x).pow(2).mean()
    loss.backward()
    print("ok")

