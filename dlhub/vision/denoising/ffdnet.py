
import torch
from torch import nn

from ._utils import pad_to_multiple, unpad


class _FFDNetBackbone(nn.Module):
    """FFDNet backbone that predicts residual noise in pixel-unshuffled space."""

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        features: int,
        depth: int,
        use_bn: bool = True,
    ) -> None:
        super().__init__()
        c_in = int(in_channels)
        c_out = int(out_channels)
        f = int(features)
        d = int(depth)
        if d < 3:
            raise ValueError("FFDNet depth must be >= 3")
        if c_in <= 0:
            raise ValueError("in_channels must be > 0")
        if c_out <= 0:
            raise ValueError("out_channels must be > 0")

        layers: list[nn.Module] = []
        layers.append(nn.Conv2d(c_in, f, kernel_size=3, padding=1, bias=True))
        layers.append(nn.ReLU(inplace=True))

        for _ in range(d - 2):
            layers.append(nn.Conv2d(f, f, kernel_size=3, padding=1, bias=not bool(use_bn)))
            if use_bn:
                layers.append(nn.BatchNorm2d(f))
            layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(f, c_out, kernel_size=3, padding=1, bias=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FFDNetDenoiser(nn.Module):
    """FFDNet (Fast and Flexible Denoising CNN), torch-only toy implementation.

    FFDNet conditions on a (global) noise level by concatenating a noise map.
    This wrapper:
      - builds a fixed noise map from `sigma`
      - pixel-unshuffles the input by 2x (space-to-depth)
      - predicts residual noise and subtracts it from the input
    """

    def __init__(
        self,
        *,
        in_channels: int,
        sigma: float = 0.1,
        features: int = 64,
        depth: int = 12,
        use_bn: bool = True,
    ) -> None:
        super().__init__()
        c = int(in_channels)
        if c <= 0:
            raise ValueError("in_channels must be > 0")

        self.in_channels = c
        self.sigma = float(sigma)
        self.unshuffle = nn.PixelUnshuffle(2)
        self.shuffle = nn.PixelShuffle(2)

        # After PixelUnshuffle(2): C -> C*4. Add noise map (1 -> 1*4).
        backbone_in = c * 4 + 4
        self.backbone = _FFDNetBackbone(
            in_channels=int(backbone_in),
            out_channels=int(c * 4),
            features=int(features),
            depth=int(depth),
            use_bn=bool(use_bn),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        if x.ndim != 4:
            raise ValueError(f"Expected input shape (B, C, H, W), got {tuple(x.shape)}")
        if int(x.shape[1]) != int(self.in_channels):
            raise ValueError(f"Expected C={self.in_channels}, got {int(x.shape[1])}")

        x_pad, pad_hw = pad_to_multiple(x, 2, mode="reflect")
        b, _, h, w = x_pad.shape

        noise_map = torch.full((b, 1, h, w), float(self.sigma), device=x_pad.device, dtype=x_pad.dtype)
        x_u = self.unshuffle(x_pad)
        n_u = self.unshuffle(noise_map)
        inp = torch.cat([x_u, n_u], dim=1)

        res_u = self.backbone(inp)
        res = self.shuffle(res_u)
        y = x_pad - res
        return unpad(y, pad_hw)


_VARIANTS: dict[str, dict] = {
    "ffdnet_tiny": {"features": 32, "depth": 7, "use_bn": True},
    "ffdnet_small": {"features": 48, "depth": 9, "use_bn": True},
    "ffdnet_base": {"features": 64, "depth": 12, "use_bn": True},
}


def build_ffdnet_denoiser(
    *,
    in_channels: int,
    sigma: float = 0.1,
    variant: str = "ffdnet_base",
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown FFDNet variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    return FFDNetDenoiser(
        in_channels=int(in_channels),
        sigma=float(sigma),
        features=int(spec["features"]),
        depth=int(spec["depth"]),
        use_bn=bool(spec["use_bn"]),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.rand(2, 1, 64, 64)
    noisy = (x + torch.randn_like(x) * 0.1).clamp(0.0, 1.0)
    m = build_ffdnet_denoiser(in_channels=1, sigma=0.1, variant="ffdnet_tiny")
    y = m(noisy)
    print("ffdnet_tiny", tuple(y.shape))
    loss = (y - x).pow(2).mean()
    loss.backward()
    print("ok")
