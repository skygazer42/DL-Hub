import torch
import torch.nn.functional as F
from torch import nn


class _DenseBlock(nn.Module):
    """DenseNet-style block: conv layers with concatenation + 1x1 fusion."""

    def __init__(self, in_ch: int, out_ch: int, *, num_layers: int = 4, growth: int = 16) -> None:
        super().__init__()
        c_in = int(in_ch)
        c_out = int(out_ch)
        num_layers_int = int(num_layers)
        g = int(growth)
        if c_in <= 0 or c_out <= 0:
            raise ValueError("channels must be > 0")
        if num_layers_int <= 0:
            raise ValueError("num_layers must be > 0")
        if g <= 0:
            raise ValueError("growth must be > 0")

        layers: list[nn.Module] = []
        ch = c_in
        for _ in range(num_layers_int):
            layers.append(nn.Conv2d(ch, g, kernel_size=3, padding=1, bias=True))
            ch += g
        self.layers = nn.ModuleList(layers)
        self.fuse = nn.Conv2d(ch, c_out, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats: list[torch.Tensor] = [x]
        for conv in self.layers:
            y = F.relu(conv(torch.cat(feats, dim=1)), inplace=True)
            feats.append(y)
        y = self.fuse(torch.cat(feats, dim=1))
        return F.relu(y, inplace=True)


class DenseUNet(nn.Module):
    """Dense U-Net style denoiser (compact-first, pure torch).

    Uses DenseNet-style blocks inside a U-Net encoder-decoder.
    Predicts a residual/noise map and returns `x - residual`.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        base_channels: int = 16,
        levels: int = 3,
        num_layers: int = 4,
        growth: int = 16,
    ) -> None:
        super().__init__()
        c_in = int(in_channels)
        base = int(base_channels)
        lv = int(levels)
        if c_in <= 0:
            raise ValueError("in_channels must be > 0")
        if base < 8:
            raise ValueError("base_channels must be >= 8")
        if lv < 2:
            raise ValueError("levels must be >= 2")

        channels = [base * (2**i) for i in range(lv)]

        enc_blocks: list[nn.Module] = []
        prev = c_in
        for ch in channels:
            enc_blocks.append(_DenseBlock(prev, ch, num_layers=int(num_layers), growth=int(growth)))
            prev = ch
        self.enc_blocks = nn.ModuleList(enc_blocks)
        self.pools = nn.ModuleList([nn.MaxPool2d(kernel_size=2, stride=2) for _ in range(lv - 1)])

        upconvs: list[nn.Module] = []
        dec_blocks: list[nn.Module] = []
        for in_ch, out_ch in zip(reversed(channels[1:]), reversed(channels[:-1]), strict=True):
            upconvs.append(
                nn.ConvTranspose2d(int(in_ch), int(out_ch), kernel_size=2, stride=2, bias=True)
            )
            dec_blocks.append(
                _DenseBlock(
                    int(out_ch) * 2, int(out_ch), num_layers=int(num_layers), growth=int(growth)
                )
            )
        self.upconvs = nn.ModuleList(upconvs)
        self.dec_blocks = nn.ModuleList(dec_blocks)

        self.out_conv = nn.Conv2d(int(channels[0]), c_in, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        if x.ndim != 4:
            raise ValueError(f"Expected input shape (B, C, H, W), got {tuple(x.shape)}")
        inp = x

        skips: list[torch.Tensor] = []
        for i, block in enumerate(self.enc_blocks):
            x = block(x)
            skips.append(x)
            if i < len(self.pools):
                x = self.pools[i](x)

        for up, dec, skip in zip(self.upconvs, self.dec_blocks, reversed(skips[:-1]), strict=True):
            x = up(x)
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
            x = torch.cat([skip, x], dim=1)
            x = dec(x)

        residual = self.out_conv(x)
        return inp - residual


_VARIANTS: dict[str, dict] = {
    "denseunet_tiny": {"base_channels": 12, "levels": 3, "num_layers": 3, "growth": 8},
    "denseunet_small": {"base_channels": 16, "levels": 3, "num_layers": 4, "growth": 12},
    "denseunet_base": {"base_channels": 24, "levels": 4, "num_layers": 4, "growth": 16},
}


def build_denseunet_denoiser(*, in_channels: int, variant: str = "denseunet_small") -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown DenseUNet variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    return DenseUNet(
        in_channels=int(in_channels),
        base_channels=int(spec["base_channels"]),
        levels=int(spec["levels"]),
        num_layers=int(spec["num_layers"]),
        growth=int(spec["growth"]),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.rand(2, 1, 64, 64)
    noisy = (x + torch.randn_like(x) * 0.1).clamp(0.0, 1.0)
    m = build_denseunet_denoiser(in_channels=1, variant="denseunet_tiny")
    y = m(noisy)
    print("denseunet_tiny", tuple(y.shape))
    loss = (y - x).pow(2).mean()
    loss.backward()
    print("ok")
