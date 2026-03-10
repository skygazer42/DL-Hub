import torch
import torch.nn.functional as F
from torch import nn

from ._utils import pad_to_multiple, unpad


def _to_tokens(x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
    b, c, h, w = x.shape
    t = x.flatten(2).transpose(1, 2).contiguous()  # (B, HW, C)
    return t, int(h), int(w)


def _from_tokens(t: torch.Tensor, h: int, w: int) -> torch.Tensor:
    b, seq_len, c = t.shape
    if seq_len != h * w:
        raise ValueError(f"Token length {seq_len} does not match H*W {h*w}")
    return t.transpose(1, 2).contiguous().reshape(b, c, h, w)


class TransformerBlock(nn.Module):
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

        self.norm1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(embed_dim=d, num_heads=h, batch_first=True)
        self.norm2 = nn.LayerNorm(d)

        hidden = max(8, int(d * float(mlp_ratio)))
        self.mlp = nn.Sequential(
            nn.Linear(d, hidden),
            nn.GELU(),
            nn.Linear(hidden, d),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C)
        y = self.norm1(x)
        y, _ = self.attn(y, y, y, need_weights=False)
        x = x + y
        x = x + self.mlp(self.norm2(x))
        return x


class TransformerStage2D(nn.Module):
    def __init__(self, dim: int, *, depth: int, num_heads: int) -> None:
        super().__init__()
        d = int(depth)
        if d <= 0:
            raise ValueError("depth must be > 0")
        self.blocks = nn.ModuleList(
            [TransformerBlock(int(dim), num_heads=int(num_heads)) for _ in range(d)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t, h, w = _to_tokens(x)
        for blk in self.blocks:
            t = blk(t)
        return _from_tokens(t, h, w)


class UFormer(nn.Module):
    """UFormer-style U-shaped Transformer denoiser (toy-first, pure torch).

    Notes:
    - This is a simplified "U-shaped transformer" for small images.
    - It uses global MHSA on flattened tokens (no windowing) but keeps tiny dims for CPU tests.
    - Residual learning: output = input + predicted_residual.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        dims: tuple[int, int, int] = (24, 48, 96),
        depths: tuple[int, int, int] = (1, 1, 2),
        heads: tuple[int, int, int] = (3, 3, 6),
    ) -> None:
        super().__init__()
        c_in = int(in_channels)
        if c_in <= 0:
            raise ValueError("in_channels must be > 0")
        dims = tuple(int(x) for x in dims)
        depths = tuple(int(x) for x in depths)
        heads = tuple(int(x) for x in heads)
        if len(dims) != 3 or len(depths) != 3 or len(heads) != 3:
            raise ValueError("dims/depths/heads must be 3-tuples")
        if any(d < 8 for d in dims):
            raise ValueError("all dims must be >= 8")

        d0, d1, d2 = dims
        n0, n1, n2 = depths
        h0, h1, h2 = heads

        self.intro = nn.Conv2d(c_in, d0, kernel_size=3, padding=1, bias=True)

        self.enc1 = TransformerStage2D(d0, depth=n0, num_heads=h0)
        self.down1 = nn.Conv2d(d0, d1, kernel_size=2, stride=2, bias=True)
        self.enc2 = TransformerStage2D(d1, depth=n1, num_heads=h1)
        self.down2 = nn.Conv2d(d1, d2, kernel_size=2, stride=2, bias=True)

        self.bott = TransformerStage2D(d2, depth=n2, num_heads=h2)

        self.up2 = nn.Conv2d(d2, d1, kernel_size=1, bias=True)
        self.reduce2 = nn.Conv2d(d1 + d1, d1, kernel_size=1, bias=True)
        self.dec2 = TransformerStage2D(d1, depth=max(1, n1), num_heads=h1)

        self.up1 = nn.Conv2d(d1, d0, kernel_size=1, bias=True)
        self.reduce1 = nn.Conv2d(d0 + d0, d0, kernel_size=1, bias=True)
        self.dec1 = TransformerStage2D(d0, depth=max(1, n0), num_heads=h0)

        self.outro = nn.Conv2d(d0, c_in, kernel_size=3, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        if x.ndim != 4:
            raise ValueError(f"Expected input shape (B, C, H, W), got {tuple(x.shape)}")

        x_pad, pad_hw = pad_to_multiple(x, 4, mode="reflect")
        inp = x_pad

        x1 = self.enc1(self.intro(x_pad))
        x2 = self.enc2(self.down1(x1))
        x3 = self.bott(self.down2(x2))

        u2 = F.interpolate(x3, scale_factor=2, mode="nearest")
        u2 = self.up2(u2)
        u2 = self.dec2(self.reduce2(torch.cat([u2, x2], dim=1)))

        u1 = F.interpolate(u2, scale_factor=2, mode="nearest")
        u1 = self.up1(u1)
        u1 = self.dec1(self.reduce1(torch.cat([u1, x1], dim=1)))

        y = inp + self.outro(u1)
        return unpad(y, pad_hw)


_VARIANTS: dict[str, dict] = {
    "uformer_tiny": {"dims": (24, 48, 96), "depths": (1, 1, 2), "heads": (3, 3, 6)},
    "uformer_small": {"dims": (32, 64, 128), "depths": (1, 2, 2), "heads": (4, 4, 8)},
    "uformer_base": {"dims": (48, 96, 192), "depths": (2, 2, 3), "heads": (6, 6, 12)},
}


def build_uformer_denoiser(
    *,
    in_channels: int,
    variant: str = "uformer_small",
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown UFormer variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    return UFormer(
        in_channels=int(in_channels),
        dims=tuple(map(int, spec["dims"])),
        depths=tuple(map(int, spec["depths"])),
        heads=tuple(map(int, spec["heads"])),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.rand(2, 1, 32, 32)
    noisy = (x + torch.randn_like(x) * 0.1).clamp(0.0, 1.0)
    m = build_uformer_denoiser(in_channels=1, variant="uformer_tiny")
    y = m(noisy)
    print("uformer_tiny", tuple(y.shape))
    loss = (y - x).pow(2).mean()
    loss.backward()
    print("ok")
