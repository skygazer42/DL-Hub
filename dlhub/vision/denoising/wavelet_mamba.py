import torch
from torch import nn

from dlhub.vision.backbones._blocks import LayerNorm2d

from ._utils import pad_to_multiple, unpad


class SimpleGate(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    """WaveletMamba block (Nonlinear Activation Free), toy-first implementation."""

    def __init__(
        self,
        channels: int,
        *,
        dw_expand: int = 2,
        ffn_expand: int = 2,
    ) -> None:
        super().__init__()
        c = int(channels)
        if c <= 0:
            raise ValueError("channels must be > 0")

        dw_ch = c * int(dw_expand)
        ffn_ch = c * int(ffn_expand)

        self.norm1 = LayerNorm2d(c)
        self.pw1 = nn.Conv2d(c, dw_ch * 2, kernel_size=1, bias=True)
        self.dwconv = nn.Conv2d(
            dw_ch * 2, dw_ch * 2, kernel_size=3, padding=1, groups=dw_ch * 2, bias=True
        )
        self.sg = SimpleGate()

        self.sca_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.sca = nn.Conv2d(dw_ch, dw_ch, kernel_size=1, bias=True)
        self.sca_gate = nn.Sigmoid()

        self.pw2 = nn.Conv2d(dw_ch, c, kernel_size=1, bias=True)
        self.beta = nn.Parameter(torch.zeros(1, c, 1, 1))

        self.norm2 = LayerNorm2d(c)
        self.ffn1 = nn.Conv2d(c, ffn_ch * 2, kernel_size=1, bias=True)
        self.ffn2 = nn.Conv2d(ffn_ch, c, kernel_size=1, bias=True)
        self.gamma = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm1(x)
        y = self.pw1(y)
        y = self.dwconv(y)
        y = self.sg(y)  # (B, dw_ch, H, W)
        ca = self.sca_gate(self.sca(self.sca_pool(y)))
        y = y * ca
        y = self.pw2(y)
        x = x + y * self.beta

        y = self.norm2(x)
        y = self.ffn1(y)
        y = self.sg(y)  # (B, ffn_ch, H, W)
        y = self.ffn2(y)
        return x + y * self.gamma


class WaveletMamba(nn.Module):
    """WaveletMamba-style encoder/decoder denoiser (pure torch, toy-first)."""

    def __init__(
        self,
        *,
        in_channels: int,
        width: int = 24,
        depths: tuple[int, int, int, int] = (1, 1, 2, 2),
        dw_expand: int = 2,
        ffn_expand: int = 2,
    ) -> None:
        super().__init__()
        c_in = int(in_channels)
        w0 = int(width)
        if c_in <= 0:
            raise ValueError("in_channels must be > 0")
        if w0 < 8:
            raise ValueError("width must be >= 8")
        depths = tuple(int(x) for x in depths)
        if len(depths) != 4:
            raise ValueError("depths must be a 4-tuple")

        dims = (w0, w0 * 2, w0 * 4, w0 * 8)

        self.intro = nn.Conv2d(c_in, dims[0], kernel_size=3, padding=1, bias=True)

        def make_stage(ch: int, depth: int) -> nn.Sequential:
            return nn.Sequential(
                *[
                    NAFBlock(ch, dw_expand=int(dw_expand), ffn_expand=int(ffn_expand))
                    for _ in range(int(depth))
                ]
            )

        self.enc1 = make_stage(dims[0], depths[0])
        self.down1 = nn.Conv2d(dims[0], dims[1], kernel_size=2, stride=2, bias=True)
        self.enc2 = make_stage(dims[1], depths[1])
        self.down2 = nn.Conv2d(dims[1], dims[2], kernel_size=2, stride=2, bias=True)
        self.enc3 = make_stage(dims[2], depths[2])
        self.down3 = nn.Conv2d(dims[2], dims[3], kernel_size=2, stride=2, bias=True)
        self.bottleneck = make_stage(dims[3], depths[3])

        self.up3 = nn.Sequential(
            nn.Conv2d(dims[3], dims[2] * 4, kernel_size=1, bias=True), nn.PixelShuffle(2)
        )
        self.reduce3 = nn.Conv2d(dims[2] * 2, dims[2], kernel_size=1, bias=True)
        self.dec3 = make_stage(dims[2], max(1, depths[2] // 2))

        self.up2 = nn.Sequential(
            nn.Conv2d(dims[2], dims[1] * 4, kernel_size=1, bias=True), nn.PixelShuffle(2)
        )
        self.reduce2 = nn.Conv2d(dims[1] * 2, dims[1], kernel_size=1, bias=True)
        self.dec2 = make_stage(dims[1], max(1, depths[1] // 2))

        self.up1 = nn.Sequential(
            nn.Conv2d(dims[1], dims[0] * 4, kernel_size=1, bias=True), nn.PixelShuffle(2)
        )
        self.reduce1 = nn.Conv2d(dims[0] * 2, dims[0], kernel_size=1, bias=True)
        self.dec1 = make_stage(dims[0], max(1, depths[0] // 2))

        self.outro = nn.Conv2d(dims[0], c_in, kernel_size=3, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        if x.ndim != 4:
            raise ValueError(f"Expected input shape (B, C, H, W), got {tuple(x.shape)}")

        x_pad, pad_hw = pad_to_multiple(x, 8, mode="reflect")
        inp = x_pad

        x1 = self.enc1(self.intro(x_pad))
        x2 = self.enc2(self.down1(x1))
        x3 = self.enc3(self.down2(x2))
        x4 = self.bottleneck(self.down3(x3))

        x = self.up3(x4)
        x = self.dec3(self.reduce3(torch.cat([x, x3], dim=1)))

        x = self.up2(x)
        x = self.dec2(self.reduce2(torch.cat([x, x2], dim=1)))

        x = self.up1(x)
        x = self.dec1(self.reduce1(torch.cat([x, x1], dim=1)))

        y = inp + self.outro(x)
        return unpad(y, pad_hw)


_VARIANTS: dict[str, dict] = {
    "wavelet_mamba_tiny": {"width": 20, "depths": (1, 1, 1, 2), "dw": 2, "ffn": 2},
    "wavelet_mamba_small": {"width": 24, "depths": (1, 1, 2, 2), "dw": 2, "ffn": 2},
    "wavelet_mamba_base": {"width": 32, "depths": (2, 2, 3, 4), "dw": 2, "ffn": 2},
}


def build_wavelet_mamba_denoiser(
    *,
    in_channels: int,
    variant: str = "wavelet_mamba_small",
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown WaveletMamba variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    return WaveletMamba(
        in_channels=int(in_channels),
        width=int(spec["width"]),
        depths=tuple(map(int, spec["depths"])),
        dw_expand=int(spec["dw"]),
        ffn_expand=int(spec["ffn"]),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 1, 64, 64)
    m = build_wavelet_mamba_denoiser(in_channels=1, variant="wavelet_mamba_tiny")
    y = m(x)
    print("wavelet_mamba_tiny", tuple(y.shape))
    loss = (y - x).pow(2).mean()
    loss.backward()
    print("ok")

