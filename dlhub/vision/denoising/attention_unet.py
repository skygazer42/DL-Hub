import torch
import torch.nn.functional as F
from torch import nn


class _DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(int(in_ch), int(out_ch), kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(int(out_ch)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(out_ch), int(out_ch), kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(int(out_ch)),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = _DoubleConv(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class _AttentionGate(nn.Module):
    """Attention gate for skip connections (Attention U-Net)."""

    def __init__(self, f_g: int, f_l: int, f_int: int) -> None:
        super().__init__()
        self.w_g = nn.Conv2d(int(f_g), int(f_int), kernel_size=1, bias=True)
        self.w_x = nn.Conv2d(int(f_l), int(f_int), kernel_size=1, bias=True)
        self.psi = nn.Conv2d(int(f_int), 1, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        # x: skip features, g: gating features
        if x.shape[-2:] != g.shape[-2:]:
            g = F.interpolate(g, size=x.shape[-2:], mode="nearest")
        a = F.relu(self.w_g(g) + self.w_x(x), inplace=True)
        psi = torch.sigmoid(self.psi(a))
        return x * psi


class _Up(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(int(in_ch), int(out_ch), kernel_size=2, stride=2)
        self.att = _AttentionGate(f_g=int(out_ch), f_l=int(out_ch), f_int=max(4, int(out_ch) // 2))
        self.conv = _DoubleConv(int(out_ch) * 2, int(out_ch))

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
        skip = self.att(skip, x)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class AttentionUNet(nn.Module):
    """Attention U-Net denoiser (compact-first, pure torch).

    Encoder-decoder with attention-gated skip connections. Predicts a residual/noise map
    and returns `x - residual`.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        base_channels: int = 32,
        levels: int = 4,
    ) -> None:
        super().__init__()
        c_in = int(in_channels)
        base = int(base_channels)
        lv = int(levels)
        if lv < 2:
            raise ValueError("levels must be >= 2")

        self.inc = _DoubleConv(c_in, base)

        downs: list[nn.Module] = []
        ch = base
        for _ in range(lv - 1):
            downs.append(_Down(ch, ch * 2))
            ch *= 2
        self.downs = nn.ModuleList(downs)

        ups: list[nn.Module] = []
        for _ in range(lv - 1):
            ups.append(_Up(ch, ch // 2))
            ch //= 2
        self.ups = nn.ModuleList(ups)

        self.outc = nn.Conv2d(base, c_in, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        if x.ndim != 4:
            raise ValueError(f"Expected input shape (B, C, H, W), got {tuple(x.shape)}")
        inp = x

        skips: list[torch.Tensor] = []
        x = self.inc(x)
        skips.append(x)
        for down in self.downs:
            x = down(x)
            skips.append(x)

        for up, skip in zip(self.ups, reversed(skips[:-1]), strict=True):
            x = up(x, skip)

        residual = self.outc(x)
        return inp - residual


_VARIANTS: dict[str, dict] = {
    "attention_unet_tiny": {"base_channels": 16, "levels": 3},
    "attention_unet_small": {"base_channels": 24, "levels": 4},
    "attention_unet_base": {"base_channels": 32, "levels": 4},
}


def build_attention_unet_denoiser(
    *, in_channels: int, variant: str = "attention_unet_small"
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(
            f"Unknown AttentionUNet variant: {variant!r}. Supported: {sorted(_VARIANTS)}"
        )
    spec = _VARIANTS[name]
    return AttentionUNet(
        in_channels=int(in_channels),
        base_channels=int(spec["base_channels"]),
        levels=int(spec["levels"]),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.rand(2, 1, 64, 64)
    noisy = (x + torch.randn_like(x) * 0.1).clamp(0.0, 1.0)
    m = build_attention_unet_denoiser(in_channels=1, variant="attention_unet_tiny")
    y = m(noisy)
    print("attention_unet_tiny", tuple(y.shape))
    loss = (y - x).pow(2).mean()
    loss.backward()
    print("ok")
