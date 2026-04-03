import torch
from torch import nn

from ._utils import pad_to_multiple, unpad


class _TokenMixer(nn.Module):
    def __init__(self, dim: int, *, num_heads: int, mlp_ratio: float = 2.0) -> None:
        super().__init__()
        d = int(dim)
        h = int(num_heads)
        if d <= 0:
            raise ValueError("dim must be > 0")
        if h <= 0:
            raise ValueError("num_heads must be > 0")
        if d % h != 0:
            raise ValueError(f"dim ({d}) must be divisible by num_heads ({h})")

        hidden = max(8, int(round(d * float(mlp_ratio))))
        self.norm1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, h, batch_first=True)
        self.norm2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(
            nn.Linear(d, hidden),
            nn.GELU(),
            nn.Linear(hidden, d),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm1(x)
        y, _ = self.attn(y, y, y, need_weights=False)
        x = x + y
        return x + self.mlp(self.norm2(x))


class _WeatherBlock(nn.Module):
    def __init__(self, dim: int, *, num_heads: int, mlp_ratio: float = 2.0) -> None:
        super().__init__()
        d = int(dim)
        self.local = nn.Sequential(
            nn.Conv2d(d, d, kernel_size=3, padding=1, groups=d, bias=True),
            nn.GELU(),
            nn.Conv2d(d, d, kernel_size=1, bias=True),
        )
        self.mixer = _TokenMixer(d, num_heads=int(num_heads), mlp_ratio=float(mlp_ratio))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x = x + self.local(x)
        tokens = x.flatten(2).transpose(1, 2).contiguous()
        tokens = self.mixer(tokens)
        return tokens.transpose(1, 2).contiguous().reshape(b, c, h, w)


class TransWeather(nn.Module):
    """Compact TransWeather-style derainer for toy data and CPU tests."""

    def __init__(
        self,
        *,
        in_channels: int,
        dim: int = 24,
        depth: int = 2,
        num_heads: int = 3,
        mlp_ratio: float = 2.0,
    ) -> None:
        super().__init__()
        c_in = int(in_channels)
        d = int(dim)
        n = int(depth)
        if c_in <= 0:
            raise ValueError("in_channels must be > 0")
        if d < 8:
            raise ValueError("dim must be >= 8")
        if n <= 0:
            raise ValueError("depth must be > 0")

        self.in_channels = c_in
        self.stem = nn.Conv2d(c_in, d, kernel_size=3, padding=1, bias=True)
        self.encoder = nn.ModuleList(
            [_WeatherBlock(d, num_heads=int(num_heads), mlp_ratio=float(mlp_ratio)) for _ in range(n)]
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(d, d, kernel_size=3, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(d, c_in, kernel_size=3, padding=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        if x.ndim != 4:
            raise ValueError(f"Expected input shape (B, C, H, W), got {tuple(x.shape)}")
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels, received {int(x.shape[1])}"
            )

        x_pad, pad_hw = pad_to_multiple(x, 4, mode="reflect")
        feat = self.stem(x_pad)
        for block in self.encoder:
            feat = block(feat)
        out = x_pad + self.fuse(feat)
        return unpad(out, pad_hw)


_VARIANTS: dict[str, dict[str, float | int]] = {
    "transweather_tiny": {"dim": 16, "depth": 1, "num_heads": 2, "mlp_ratio": 2.0},
    "transweather_small": {"dim": 24, "depth": 2, "num_heads": 3, "mlp_ratio": 2.0},
    "transweather_base": {"dim": 32, "depth": 3, "num_heads": 4, "mlp_ratio": 2.5},
}


def build_transweather_denoiser(
    *,
    in_channels: int,
    variant: str = "transweather_small",
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(
            f"Unknown TransWeather variant: {variant!r}. Supported: {sorted(_VARIANTS)}"
        )
    spec = _VARIANTS[name]
    return TransWeather(
        in_channels=int(in_channels),
        dim=int(spec["dim"]),
        depth=int(spec["depth"]),
        num_heads=int(spec["num_heads"]),
        mlp_ratio=float(spec["mlp_ratio"]),
    )
