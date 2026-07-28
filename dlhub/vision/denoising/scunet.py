import math

import torch
import torch.nn.functional as F
from torch import nn

from ._utils import pad_to_multiple, unpad


class _ResBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        c = int(channels)
        self.conv1 = nn.Conv2d(c, c, kernel_size=3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(c, c, kernel_size=3, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.conv1(x), inplace=True)
        y = self.conv2(y)
        return F.relu(x + y, inplace=True)


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        d = int(dim)
        h = int(hidden_dim)
        self.fc1 = nn.Linear(d, h)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(h, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class WindowAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        d = int(dim)
        h = int(num_heads)
        if d % h != 0:
            raise ValueError("dim must be divisible by num_heads")
        self.dim = d
        self.num_heads = h
        self.head_dim = d // h
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(d, 3 * d, bias=True)
        self.proj = nn.Linear(d, d, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B*nW, N, D)
        b, n, d = x.shape
        if d != self.dim:
            raise ValueError(f"Expected dim={self.dim}, got {d}")
        qkv = self.qkv(x).view(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, H, N, Dh)
        attn = torch.matmul(q, k.transpose(-2, -1)) * float(self.scale)
        attn = torch.softmax(attn, dim=-1)
        y = torch.matmul(attn, v).transpose(1, 2).contiguous().view(b, n, d)
        return self.proj(y)


def _window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    # x: (B, H, W, C) -> (B*nW, ws*ws, C)
    b, h, w, c = x.shape
    ws = int(window_size)
    x = x.view(b, h // ws, ws, w // ws, ws, c)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return x.view(b * (h // ws) * (w // ws), ws * ws, c)


def _window_reverse(
    windows: torch.Tensor, window_size: int, *, b: int, h: int, w: int, c: int
) -> torch.Tensor:
    # windows: (B*nW, ws*ws, C) -> (B, H, W, C)
    ws = int(window_size)
    x = windows.view(b, h // ws, w // ws, ws, ws, c)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return x.view(b, h, w, c)


class SwinBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        window_size: int = 4,
        shift: bool = False,
        mlp_ratio: float = 2.0,
    ) -> None:
        super().__init__()
        d = int(dim)
        self.window_size = int(window_size)
        if self.window_size <= 0:
            raise ValueError("window_size must be > 0")
        self.shift_size = self.window_size // 2 if bool(shift) else 0

        self.norm1 = nn.LayerNorm(d)
        self.attn = WindowAttention(d, int(num_heads))
        self.norm2 = nn.LayerNorm(d)
        self.mlp = MLP(d, int(round(d * float(mlp_ratio))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, H, W, C)
        b, h, w, c = x.shape
        ws = int(self.window_size)
        if h % ws != 0 or w % ws != 0:
            raise ValueError("Input must be padded so H and W are multiples of window_size")

        shortcut = x
        if self.shift_size:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

        x_w = _window_partition(x, ws)  # (B*nW, N, C)
        x_w = self.attn(self.norm1(x_w))
        x = _window_reverse(x_w, ws, b=b, h=h, w=w, c=c)

        if self.shift_size:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x


def _lcm(a: int, b: int) -> int:
    return abs(a * b) // math.gcd(a, b)


class SCUNet(nn.Module):
    """SCUNet-style hybrid Conv+Swin U-Net denoiser (compact-first, pure torch).

    This is a simplified SCUNet-inspired model:
    - Conv encoder/decoder with residual blocks
    - Swin-style window attention blocks at the bottleneck
    - Predicts a residual/noise map and returns `x - residual`
    """

    def __init__(
        self,
        *,
        in_channels: int,
        base_channels: int = 32,
        levels: int = 4,
        attn_depth: int = 2,
        num_heads: int = 4,
        window_size: int = 4,
    ) -> None:
        super().__init__()
        c_in = int(in_channels)
        base = int(base_channels)
        lv = int(levels)
        ad = int(attn_depth)
        heads = int(num_heads)
        ws = int(window_size)
        if c_in <= 0:
            raise ValueError("in_channels must be > 0")
        if base < 8:
            raise ValueError("base_channels must be >= 8")
        if lv < 2 or lv > 5:
            raise ValueError("levels must be in [2, 5]")
        if ad <= 0:
            raise ValueError("attn_depth must be > 0")
        if ws <= 0:
            raise ValueError("window_size must be > 0")

        self.window_size = ws
        self.down_factor = 2 ** (lv - 1)
        self.pad_multiple = _lcm(ws, self.down_factor)

        dims = [base * (2**i) for i in range(lv)]

        self.intro = nn.Conv2d(c_in, dims[0], kernel_size=3, padding=1, bias=True)
        self.enc0 = nn.Sequential(_ResBlock(dims[0]), _ResBlock(dims[0]))

        self.downs = nn.ModuleList()
        self.encs = nn.ModuleList()
        for i in range(lv - 1):
            self.downs.append(nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2, bias=True))
            self.encs.append(nn.Sequential(_ResBlock(dims[i + 1]), _ResBlock(dims[i + 1])))

        bottleneck_dim = dims[-1]
        if bottleneck_dim % heads != 0:
            raise ValueError("bottleneck channels must be divisible by num_heads")
        self.attn = nn.ModuleList(
            [
                SwinBlock(
                    bottleneck_dim,
                    heads,
                    window_size=ws,
                    shift=bool(i % 2 == 1),
                    mlp_ratio=2.0,
                )
                for i in range(ad)
            ]
        )
        self.bottleneck_conv = nn.Conv2d(
            bottleneck_dim, bottleneck_dim, kernel_size=3, padding=1, bias=True
        )

        self.ups = nn.ModuleList()
        self.decs = nn.ModuleList()
        for i in range(lv - 1, 0, -1):
            self.ups.append(
                nn.ConvTranspose2d(dims[i], dims[i - 1], kernel_size=2, stride=2, bias=True)
            )
            self.decs.append(nn.Sequential(_ResBlock(dims[i - 1]), _ResBlock(dims[i - 1])))

        self.outro = nn.Conv2d(dims[0], c_in, kernel_size=3, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        if x.ndim != 4:
            raise ValueError(f"Expected input shape (B, C, H, W), got {tuple(x.shape)}")

        x_pad, pad_hw = pad_to_multiple(x, self.pad_multiple, mode="reflect")

        y = F.relu(self.intro(x_pad), inplace=True)
        y = self.enc0(y)
        skips: list[torch.Tensor] = [y]
        for down, enc in zip(self.downs, self.encs, strict=True):
            y = F.relu(down(y), inplace=True)
            y = enc(y)
            skips.append(y)

        # Bottleneck window attention (operate in NHWC for LayerNorm).
        b, c, h, w = y.shape
        y_nhwc = y.permute(0, 2, 3, 1).contiguous()
        for blk in self.attn:
            y_nhwc = blk(y_nhwc)
        y = y_nhwc.permute(0, 3, 1, 2).contiguous()
        y = F.relu(self.bottleneck_conv(y), inplace=True)

        # Decode with additive skips (cheap and stable for compact use).
        for up, dec, skip in zip(self.ups, self.decs, reversed(skips[:-1]), strict=True):
            y = up(y)
            if y.shape[-2:] != skip.shape[-2:]:
                y = F.interpolate(y, size=skip.shape[-2:], mode="nearest")
            y = y + skip
            y = dec(y)

        residual = self.outro(y)
        out = x_pad - residual
        return unpad(out, pad_hw)


_VARIANTS: dict[str, dict] = {
    "scunet_tiny": {"base_channels": 24, "levels": 4, "attn_depth": 1, "heads": 4, "window": 4},
    "scunet_small": {"base_channels": 32, "levels": 4, "attn_depth": 2, "heads": 4, "window": 4},
    "scunet_base": {"base_channels": 40, "levels": 4, "attn_depth": 4, "heads": 5, "window": 4},
}


def build_scunet_denoiser(*, in_channels: int, variant: str = "scunet_small") -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown SCUNet variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    return SCUNet(
        in_channels=int(in_channels),
        base_channels=int(spec["base_channels"]),
        levels=int(spec["levels"]),
        attn_depth=int(spec["attn_depth"]),
        num_heads=int(spec["heads"]),
        window_size=int(spec["window"]),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.rand(2, 1, 64, 64)
    noisy = (x + torch.randn_like(x) * 0.1).clamp(0.0, 1.0)
    m = build_scunet_denoiser(in_channels=1, variant="scunet_tiny")
    y = m(noisy)
    print("scunet_tiny", tuple(y.shape))
    loss = (y - x).pow(2).mean()
    loss.backward()
    print("ok")
