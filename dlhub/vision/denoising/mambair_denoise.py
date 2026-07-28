import torch
from torch import nn


class MambairDenoise(nn.Module):
    """MambairDenoise (Denoising CNN), implemented from scratch in torch.

    Notes:
    - The core MambairDenoise predicts the noise/residual. A wrapper `MambairDenoiseDenoiser` converts it
      to a denoised image by subtracting the predicted residual from the input.
    - This implementation is small and training-friendly for synthetic datasets.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        features: int = 64,
        depth: int = 17,
        use_bn: bool = True,
    ) -> None:
        super().__init__()
        c_in = int(in_channels)
        f = int(features)
        d = int(depth)
        if d < 3:
            raise ValueError(f"MambairDenoise depth must be >= 3, got: {depth}")

        layers: list[nn.Module] = []
        layers.append(nn.Conv2d(c_in, f, kernel_size=3, padding=1, bias=True))
        layers.append(nn.ReLU(inplace=True))

        for _ in range(d - 2):
            layers.append(nn.Conv2d(f, f, kernel_size=3, padding=1, bias=not bool(use_bn)))
            if use_bn:
                layers.append(nn.BatchNorm2d(f))
            layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(f, c_in, kernel_size=3, padding=1, bias=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        if x.ndim != 4:
            raise ValueError(f"Expected input shape (B, C, H, W), got {tuple(x.shape)}")
        return self.net(x)


class MambairDenoiseDenoiser(nn.Module):
    """MambairDenoise wrapper that returns the denoised image instead of the residual/noise."""

    def __init__(self, backbone: MambairDenoise, *, residual: bool = True) -> None:
        super().__init__()
        self.backbone = backbone
        self.residual = bool(residual)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        noise = self.backbone(x)
        if self.residual:
            return x.to(noise.dtype) - noise
        return noise


_VARIANTS: dict[str, dict] = {
    # Canonical-ish depths (compact-friendly: keep features modest).
    "mambair_denoise_9": {"features": 48, "depth": 9, "use_bn": True},
    "mambair_denoise_17": {"features": 64, "depth": 17, "use_bn": True},
    "mambair_denoise_20": {"features": 64, "depth": 20, "use_bn": True},
    # Very small (fast smoke / CPU-friendly).
    "mambair_denoise_tiny": {"features": 32, "depth": 7, "use_bn": True},
}


def build_mambair_denoise_denoiser(
    *,
    in_channels: int,
    variant: str = "mambair_denoise_17",
    residual: bool = True,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(
            f"Unknown MambairDenoise variant: {variant!r}. Supported: {sorted(_VARIANTS)}"
        )
    spec = _VARIANTS[name]
    return MambairDenoiseDenoiser(
        MambairDenoise(
            in_channels=int(in_channels),
            features=int(spec["features"]),
            depth=int(spec["depth"]),
            use_bn=bool(spec["use_bn"]),
        ),
        residual=bool(residual),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_mambair_denoise_denoiser(in_channels=3, variant="mambair_denoise_9")
    y = m(x)
    print("mambair_denoise_9", tuple(y.shape))
    loss = (y - x).pow(2).mean()
    loss.backward()
    print("ok")
