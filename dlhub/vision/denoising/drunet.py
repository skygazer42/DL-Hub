import torch
import torch.nn.functional as F
from torch import nn

from ._utils import pad_to_multiple, unpad


def _group_norm(ch: int) -> nn.GroupNorm:
    c = int(ch)
    for g in (8, 4, 2, 1):
        if c % g == 0:
            return nn.GroupNorm(g, c)
    return nn.GroupNorm(1, c)


class ResidualBlockGN(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        c = int(channels)
        self.conv1 = nn.Conv2d(c, c, kernel_size=3, padding=1, bias=False)
        self.gn1 = _group_norm(c)
        self.conv2 = nn.Conv2d(c, c, kernel_size=3, padding=1, bias=False)
        self.gn2 = _group_norm(c)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act(self.gn1(self.conv1(x)))
        y = self.gn2(self.conv2(y))
        return self.act(x + y)


class DRUNet(nn.Module):
    """DRUNet-style denoiser (compact-first, pure torch).

    DRUNet conditions on a noise-level map. This implementation stores a fixed `sigma`
    and concatenates a (B, 1, H, W) map to the input.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        sigma: float = 0.1,
        base_channels: int = 48,
        levels: int = 4,
        blocks_per_level: int = 2,
    ) -> None:
        super().__init__()
        c_in = int(in_channels)
        if c_in <= 0:
            raise ValueError("in_channels must be > 0")
        lv = int(levels)
        if lv < 2:
            raise ValueError("levels must be >= 2")

        self.in_channels = c_in
        self.sigma = float(sigma)
        base = int(base_channels)
        bpl = int(blocks_per_level)
        if base < 8:
            raise ValueError("base_channels must be >= 8")
        if bpl <= 0:
            raise ValueError("blocks_per_level must be > 0")

        self.in_conv = nn.Conv2d(c_in + 1, base, kernel_size=3, padding=1, bias=True)

        self.enc_blocks = nn.ModuleList()
        self.downs = nn.ModuleList()
        ch = base
        for _ in range(lv - 1):
            self.enc_blocks.append(nn.Sequential(*[ResidualBlockGN(ch) for _ in range(bpl)]))
            self.downs.append(nn.Conv2d(ch, ch * 2, kernel_size=3, stride=2, padding=1, bias=True))
            ch *= 2

        self.bottleneck = nn.Sequential(*[ResidualBlockGN(ch) for _ in range(bpl)])

        self.ups = nn.ModuleList()
        self.reduces = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        for _ in range(lv - 1):
            self.ups.append(nn.ConvTranspose2d(ch, ch // 2, kernel_size=2, stride=2))
            ch //= 2
            self.reduces.append(nn.Conv2d(ch * 2, ch, kernel_size=1, bias=True))
            self.dec_blocks.append(nn.Sequential(*[ResidualBlockGN(ch) for _ in range(bpl)]))

        self.out_conv = nn.Conv2d(base, c_in, kernel_size=3, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        if x.ndim != 4:
            raise ValueError(f"Expected input shape (B, C, H, W), got {tuple(x.shape)}")
        if int(x.shape[1]) != int(self.in_channels):
            raise ValueError(f"Expected C={self.in_channels}, got {int(x.shape[1])}")

        # Levels downsample by 2^(levels-1).
        mult = 2 ** (len(self.enc_blocks))
        x_pad, pad_hw = pad_to_multiple(x, mult, mode="reflect")
        b, _, h, w = x_pad.shape

        noise_map = torch.full(
            (b, 1, h, w), float(self.sigma), device=x_pad.device, dtype=x_pad.dtype
        )
        h0 = self.in_conv(torch.cat([x_pad, noise_map], dim=1))

        skips: list[torch.Tensor] = []
        h = h0
        for blk, down in zip(self.enc_blocks, self.downs, strict=True):
            h = blk(h)
            skips.append(h)
            h = down(h)

        h = self.bottleneck(h)

        for up, reduce, blk, skip in zip(
            self.ups, self.reduces, self.dec_blocks, reversed(skips), strict=True
        ):
            h = up(h)
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode="nearest")
            h = blk(reduce(torch.cat([h, skip], dim=1)))

        y = x_pad + self.out_conv(h)
        return unpad(y, pad_hw)


_VARIANTS: dict[str, dict] = {
    "drunet_tiny": {"base": 24, "levels": 3, "blocks": 1},
    "drunet_small": {"base": 32, "levels": 4, "blocks": 2},
    "drunet_base": {"base": 48, "levels": 4, "blocks": 2},
}


def build_drunet_denoiser(
    *,
    in_channels: int,
    sigma: float = 0.1,
    variant: str = "drunet_small",
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown DRUNet variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    return DRUNet(
        in_channels=int(in_channels),
        sigma=float(sigma),
        base_channels=int(spec["base"]),
        levels=int(spec["levels"]),
        blocks_per_level=int(spec["blocks"]),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.rand(2, 1, 64, 64)
    noisy = (x + torch.randn_like(x) * 0.1).clamp(0.0, 1.0)
    m = build_drunet_denoiser(in_channels=1, sigma=0.1, variant="drunet_tiny")
    y = m(noisy)
    print("drunet_tiny", tuple(y.shape))
    loss = (y - x).pow(2).mean()
    loss.backward()
    print("ok")
