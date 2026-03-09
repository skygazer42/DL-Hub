
import torch
from torch import nn
import torch.nn.functional as F


class REDNet(nn.Module):
    """REDNet (Residual Encoder-Decoder Network) adapted for denoising.

    A symmetric conv/deconv autoencoder with skip connections between mirrored layers.
    Output is `x - residual`.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        features: int = 64,
        depth: int = 10,
    ) -> None:
        super().__init__()
        c_in = int(in_channels)
        f = int(features)
        d = int(depth)
        if c_in <= 0:
            raise ValueError("in_channels must be > 0")
        if f < 8:
            raise ValueError("features must be >= 8")
        if d <= 0:
            raise ValueError("depth must be > 0")

        self.in_conv = nn.Conv2d(c_in, f, kernel_size=3, padding=1, bias=True)

        self.enc = nn.ModuleList([nn.Conv2d(f, f, kernel_size=3, padding=1, bias=True) for _ in range(d)])
        self.dec = nn.ModuleList([nn.ConvTranspose2d(f, f, kernel_size=3, padding=1, bias=True) for _ in range(d)])

        self.out_conv = nn.Conv2d(f, c_in, kernel_size=3, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        if x.ndim != 4:
            raise ValueError(f"Expected input shape (B, C, H, W), got {tuple(x.shape)}")

        y = F.relu(self.in_conv(x), inplace=True)

        skips: list[torch.Tensor] = []
        for conv in self.enc:
            y = F.relu(conv(y), inplace=True)
            skips.append(y)

        for deconv, skip in zip(self.dec, reversed(skips), strict=True):
            y = deconv(y)
            y = F.relu(y + skip, inplace=True)

        residual = self.out_conv(y)
        return x - residual


_VARIANTS: dict[str, dict] = {
    "rednet_tiny": {"features": 32, "depth": 5},
    "rednet_small": {"features": 48, "depth": 8},
    "rednet_base": {"features": 64, "depth": 12},
}


def build_rednet_denoiser(*, in_channels: int, variant: str = "rednet_small") -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown REDNet variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    return REDNet(in_channels=int(in_channels), features=int(spec["features"]), depth=int(spec["depth"]))


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.rand(2, 3, 64, 64)
    noisy = (x + torch.randn_like(x) * 0.1).clamp(0.0, 1.0)
    m = build_rednet_denoiser(in_channels=3, variant="rednet_tiny")
    y = m(noisy)
    print("rednet_tiny", tuple(y.shape))
    loss = (y - x).pow(2).mean()
    loss.backward()
    print("ok")

