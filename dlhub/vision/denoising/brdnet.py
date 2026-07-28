import torch
import torch.nn.functional as F
from torch import nn


class _ConvStack(nn.Module):
    def __init__(self, channels: int, *, depth: int, dilation: int) -> None:
        super().__init__()
        c = int(channels)
        d = int(depth)
        dil = int(dilation)
        if c <= 0:
            raise ValueError("channels must be > 0")
        if d < 2:
            raise ValueError("depth must be >= 2")
        if dil <= 0:
            raise ValueError("dilation must be > 0")

        layers: list[nn.Module] = []
        for _ in range(d):
            layers.append(
                nn.Conv2d(
                    c,
                    c,
                    kernel_size=3,
                    padding=dil,
                    dilation=dil,
                    bias=True,
                )
            )
            layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BRDNet(nn.Module):
    """BRDNet-style denoiser (compact-first, pure torch).

    BRDNet is commonly described as a dual-branch residual denoiser:
    - A normal-conv branch (local details)
    - A dilated-conv branch (wider context)
    We fuse both and predict a residual/noise map, returning `x - residual`.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        features: int = 64,
        depth: int = 5,
        dilation_a: int = 1,
        dilation_b: int = 2,
    ) -> None:
        super().__init__()
        c_in = int(in_channels)
        f = int(features)
        d = int(depth)
        if c_in <= 0:
            raise ValueError("in_channels must be > 0")
        if f < 8:
            raise ValueError("features must be >= 8")
        if d < 2:
            raise ValueError("depth must be >= 2")

        self.stem = nn.Conv2d(c_in, f, kernel_size=3, padding=1, bias=True)
        self.branch_a = _ConvStack(f, depth=d, dilation=int(dilation_a))
        self.branch_b = _ConvStack(f, depth=d, dilation=int(dilation_b))
        self.fuse = nn.Conv2d(f * 2, f, kernel_size=1, bias=True)
        self.out = nn.Conv2d(f, c_in, kernel_size=3, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        if x.ndim != 4:
            raise ValueError(f"Expected input shape (B, C, H, W), got {tuple(x.shape)}")

        feat = F.relu(self.stem(x), inplace=True)
        a = self.branch_a(feat)
        b = self.branch_b(feat)
        y = torch.cat([a, b], dim=1)
        y = F.relu(self.fuse(y), inplace=True)
        residual = self.out(y)
        return x - residual


_VARIANTS: dict[str, dict] = {
    "brdnet_tiny": {"features": 32, "depth": 3, "dilation_a": 1, "dilation_b": 2},
    "brdnet_small": {"features": 64, "depth": 4, "dilation_a": 1, "dilation_b": 2},
    "brdnet_base": {"features": 80, "depth": 5, "dilation_a": 1, "dilation_b": 3},
}


def build_brdnet_denoiser(*, in_channels: int, variant: str = "brdnet_small") -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown BRDNet variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    return BRDNet(
        in_channels=int(in_channels),
        features=int(spec["features"]),
        depth=int(spec["depth"]),
        dilation_a=int(spec["dilation_a"]),
        dilation_b=int(spec["dilation_b"]),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.rand(2, 1, 64, 64)
    noisy = (x + torch.randn_like(x) * 0.1).clamp(0.0, 1.0)
    m = build_brdnet_denoiser(in_channels=1, variant="brdnet_tiny")
    y = m(noisy)
    print("brdnet_tiny", tuple(y.shape))
    loss = (y - x).pow(2).mean()
    loss.backward()
    print("ok")
