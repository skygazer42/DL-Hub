import torch
import torch.nn.functional as F
from torch import nn


class LayerNorm2d(nn.Module):
    """Channel-wise LayerNorm for NCHW tensors."""

    def __init__(self, channels: int, eps: float = 1e-6, *, affine: bool = True) -> None:
        super().__init__()
        self.channels = int(channels)
        self.eps = float(eps)
        self.affine = bool(affine)

        if self.affine:
            self.weight = nn.Parameter(torch.ones(self.channels))
            self.bias = nn.Parameter(torch.zeros(self.channels))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected input shape (B, C, H, W), got {tuple(x.shape)}")
        # (B, H, W, C) for layer_norm on channel dim.
        y = x.permute(0, 2, 3, 1)
        y = F.layer_norm(y, (self.channels,), self.weight, self.bias, self.eps)
        return y.permute(0, 3, 1, 2)


class MDTA(nn.Module):
    """Multi-DConv Head Transposed Attention (Restormer)."""

    def __init__(self, dim: int, *, num_heads: int) -> None:
        super().__init__()
        d = int(dim)
        h = int(num_heads)
        if d % h != 0:
            raise ValueError(f"dim must be divisible by num_heads, got dim={d}, heads={h}")

        self.dim = d
        self.num_heads = h
        self.head_dim = d // h

        self.qkv = nn.Conv2d(d, d * 3, kernel_size=1, bias=True)
        self.qkv_dw = nn.Conv2d(d * 3, d * 3, kernel_size=3, padding=1, groups=d * 3, bias=True)

        self.temperature = nn.Parameter(torch.ones(h, 1, 1))
        self.proj = nn.Conv2d(d, d, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        qkv = self.qkv_dw(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        n = h * w
        q = q.reshape(b, self.num_heads, self.head_dim, n)
        k = k.reshape(b, self.num_heads, self.head_dim, n)
        v = v.reshape(b, self.num_heads, self.head_dim, n)

        # Normalize across spatial positions (transposed attention).
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = torch.matmul(q, k.transpose(-1, -2))  # (B, heads, head_dim, head_dim)
        attn = attn * self.temperature
        attn = attn.softmax(dim=-1)

        out = torch.matmul(attn, v)  # (B, heads, head_dim, N)
        out = out.reshape(b, c, h, w)
        return self.proj(out)


class GDFN(nn.Module):
    """Gated-DConv Feed-Forward Network (Restormer)."""

    def __init__(self, dim: int, *, expansion: float = 2.0) -> None:
        super().__init__()
        d = int(dim)
        hidden = int(round(d * float(expansion)))

        self.project_in = nn.Conv2d(d, hidden * 2, kernel_size=1, bias=True)
        self.dwconv = nn.Conv2d(
            hidden * 2, hidden * 2, kernel_size=3, padding=1, groups=hidden * 2, bias=True
        )
        self.project_out = nn.Conv2d(hidden, d, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dwconv(self.project_in(x))
        x1, x2 = x.chunk(2, dim=1)
        x = F.gelu(x1) * x2
        return self.project_out(x)


class RestormerBlock(nn.Module):
    def __init__(self, dim: int, *, num_heads: int, ffn_expansion: float = 2.0) -> None:
        super().__init__()
        d = int(dim)
        self.norm1 = LayerNorm2d(d)
        self.attn = MDTA(d, num_heads=int(num_heads))
        self.norm2 = LayerNorm2d(d)
        self.ffn = GDFN(d, expansion=float(ffn_expansion))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class Downsample(nn.Module):
    def __init__(self, in_dim: int) -> None:
        super().__init__()
        c = int(in_dim)
        self.conv = nn.Conv2d(c, c * 2, kernel_size=3, stride=2, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, in_dim: int) -> None:
        super().__init__()
        c = int(in_dim)
        # Upscale x2: output channels become c/2 (since PixelShuffle divides channels by 4).
        self.conv = nn.Conv2d(c, (c // 2) * 4, kernel_size=1, bias=True)
        self.ps = nn.PixelShuffle(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ps(self.conv(x))


class Restormer(nn.Module):
    """Restormer-style encoder/decoder denoiser (pure torch).

    This is a compact, educational implementation aimed at small toy images.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        dim: int = 32,
        depths: tuple[int, int, int, int] = (2, 2, 2, 2),
        heads: tuple[int, int, int, int] = (1, 2, 4, 8),
        ffn_expansion: float = 2.0,
    ) -> None:
        super().__init__()
        c_in = int(in_channels)
        d0 = int(dim)
        if d0 < 8:
            raise ValueError(f"dim must be >= 8, got: {dim}")
        depths = tuple(int(x) for x in depths)
        heads = tuple(int(x) for x in heads)
        if len(depths) != 4 or len(heads) != 4:
            raise ValueError("depths and heads must be 4-tuples (encoder stages)")

        # Ensure divisibility for each stage.
        dims = (d0, d0 * 2, d0 * 4, d0 * 8)
        for stage_dim, nh in zip(dims, heads, strict=True):
            if stage_dim % nh != 0:
                raise ValueError(f"Stage dim {stage_dim} must be divisible by heads {nh}")

        self.embed = nn.Conv2d(c_in, d0, kernel_size=3, padding=1, bias=True)

        def make_stage(stage_dim: int, depth: int, nh: int) -> nn.Sequential:
            blocks: list[nn.Module] = []
            for _ in range(int(depth)):
                blocks.append(
                    RestormerBlock(stage_dim, num_heads=int(nh), ffn_expansion=float(ffn_expansion))
                )
            return nn.Sequential(*blocks)

        self.enc1 = make_stage(dims[0], depths[0], heads[0])
        self.down1 = Downsample(dims[0])
        self.enc2 = make_stage(dims[1], depths[1], heads[1])
        self.down2 = Downsample(dims[1])
        self.enc3 = make_stage(dims[2], depths[2], heads[2])
        self.down3 = Downsample(dims[2])
        self.bottleneck = make_stage(dims[3], depths[3], heads[3])

        self.up3 = Upsample(dims[3])  # -> dims[2]
        self.reduce3 = nn.Conv2d(dims[2] * 2, dims[2], kernel_size=1, bias=True)
        self.dec3 = make_stage(dims[2], max(1, depths[2] // 2), heads[2])

        self.up2 = Upsample(dims[2])  # -> dims[1]
        self.reduce2 = nn.Conv2d(dims[1] * 2, dims[1], kernel_size=1, bias=True)
        self.dec2 = make_stage(dims[1], max(1, depths[1] // 2), heads[1])

        self.up1 = Upsample(dims[1])  # -> dims[0]
        self.reduce1 = nn.Conv2d(dims[0] * 2, dims[0], kernel_size=1, bias=True)
        self.dec1 = make_stage(dims[0], max(1, depths[0] // 2), heads[0])

        self.out = nn.Conv2d(dims[0], c_in, kernel_size=3, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        if x.ndim != 4:
            raise ValueError(f"Expected input shape (B, C, H, W), got {tuple(x.shape)}")

        inp = x
        x1 = self.enc1(self.embed(x))
        x2 = self.enc2(self.down1(x1))
        x3 = self.enc3(self.down2(x2))
        x4 = self.bottleneck(self.down3(x3))

        x = self.up3(x4)
        x = self.dec3(self.reduce3(torch.cat([x, x3], dim=1)))

        x = self.up2(x)
        x = self.dec2(self.reduce2(torch.cat([x, x2], dim=1)))

        x = self.up1(x)
        x = self.dec1(self.reduce1(torch.cat([x, x1], dim=1)))

        # Residual prediction (common for restoration).
        return inp + self.out(x)


_VARIANTS: dict[str, dict] = {
    # Keep variants CPU-friendly; toy-first.
    "restormer_tiny": {"dim": 24, "depths": (1, 1, 2, 2), "heads": (1, 2, 4, 8), "ffn": 2.0},
    "restormer_small": {"dim": 32, "depths": (2, 2, 3, 4), "heads": (1, 2, 4, 8), "ffn": 2.0},
    "restormer_base": {"dim": 48, "depths": (2, 3, 4, 6), "heads": (1, 2, 4, 8), "ffn": 2.66},
}


def build_restormer_denoiser(
    *,
    in_channels: int,
    variant: str = "restormer_tiny",
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown Restormer variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    return Restormer(
        in_channels=int(in_channels),
        dim=int(spec["dim"]),
        depths=tuple(map(int, spec["depths"])),
        heads=tuple(map(int, spec["heads"])),
        ffn_expansion=float(spec["ffn"]),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_restormer_denoiser(in_channels=3, variant="restormer_tiny")
    y = m(x)
    print("restormer_tiny", tuple(y.shape))
    loss = (y - x).pow(2).mean()
    loss.backward()
    print("ok")
