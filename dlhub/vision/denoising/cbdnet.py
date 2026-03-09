
import torch
from torch import nn
import torch.nn.functional as F

from ._utils import pad_to_multiple, unpad


class _ConvAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, *, k: int = 3, act: bool = True) -> None:
        super().__init__()
        k = int(k)
        p = k // 2
        self.conv = nn.Conv2d(int(in_ch), int(out_ch), kernel_size=k, padding=p, bias=True)
        self.act = nn.ReLU(inplace=True) if bool(act) else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x))


class NoiseEstimationNet(nn.Module):
    """CBDNet noise estimation sub-network (toy-first).

    Produces a per-pixel noise-level map. In the original CBDNet, this is trained with
    a dedicated noise estimation loss; here we keep it end-to-end and let supervision
    flow through the final denoising loss.
    """

    def __init__(self, *, in_channels: int, width: int = 32, depth: int = 5) -> None:
        super().__init__()
        c_in = int(in_channels)
        w0 = int(width)
        d = int(depth)
        if c_in <= 0:
            raise ValueError("in_channels must be > 0")
        if w0 < 8:
            raise ValueError("width must be >= 8")
        if d < 2:
            raise ValueError("depth must be >= 2")

        layers: list[nn.Module] = [_ConvAct(c_in, w0, k=3, act=True)]
        for _ in range(d - 2):
            layers.append(_ConvAct(w0, w0, k=3, act=True))
        layers.append(_ConvAct(w0, c_in, k=3, act=False))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # map in [0,1] (interpretation depends on training)
        return torch.sigmoid(self.net(x.to(torch.float32)))


class _UNetBlock(nn.Module):
    def __init__(self, ch: int) -> None:
        super().__init__()
        c = int(ch)
        self.conv1 = _ConvAct(c, c, k=3, act=True)
        self.conv2 = _ConvAct(c, c, k=3, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv2(self.conv1(x))
        return F.relu(x + y, inplace=True)


class ConditionalDenoiserUNet(nn.Module):
    """A small conditional U-Net used by CBDNet (toy-first).

    Input is `[noisy, noise_map]` concatenated along channels.
    Output is a predicted residual/noise that gets subtracted from the input noisy image.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        width: int = 48,
        depth: int = 2,
    ) -> None:
        super().__init__()
        c_in = int(in_channels)
        w0 = int(width)
        d = int(depth)
        if c_in <= 0:
            raise ValueError("in_channels must be > 0")
        if w0 < 8:
            raise ValueError("width must be >= 8")
        if d <= 0:
            raise ValueError("depth must be > 0")

        def stage(ch: int) -> nn.Sequential:
            return nn.Sequential(*[_UNetBlock(ch) for _ in range(d)])

        self.intro = nn.Conv2d(c_in, w0, kernel_size=3, padding=1, bias=True)
        self.enc1 = stage(w0)
        self.down1 = nn.Conv2d(w0, w0 * 2, kernel_size=2, stride=2, bias=True)
        self.enc2 = stage(w0 * 2)
        self.down2 = nn.Conv2d(w0 * 2, w0 * 4, kernel_size=2, stride=2, bias=True)
        self.bott = stage(w0 * 4)

        self.up2 = nn.Conv2d(w0 * 4, w0 * 2, kernel_size=1, bias=True)
        self.reduce2 = nn.Conv2d(w0 * 4, w0 * 2, kernel_size=1, bias=True)
        self.dec2 = stage(w0 * 2)

        self.up1 = nn.Conv2d(w0 * 2, w0, kernel_size=1, bias=True)
        self.reduce1 = nn.Conv2d(w0 * 2, w0, kernel_size=1, bias=True)
        self.dec1 = stage(w0)

        # Predict residual/noise (same channels as noisy image, not the concatenated input).
        self.outro = nn.Conv2d(w0, c_in // 2, kernel_size=3, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        x1 = self.enc1(self.intro(x))
        x2 = self.enc2(self.down1(x1))
        x3 = self.bott(self.down2(x2))

        u2 = F.interpolate(x3, scale_factor=2, mode="nearest")
        u2 = self.up2(u2)
        u2 = self.dec2(self.reduce2(torch.cat([u2, x2], dim=1)))

        u1 = F.interpolate(u2, scale_factor=2, mode="nearest")
        u1 = self.up1(u1)
        u1 = self.dec1(self.reduce1(torch.cat([u1, x1], dim=1)))

        return self.outro(u1)


class CBDNet(nn.Module):
    """CBDNet-style blind denoiser (toy-first, pure torch).

    Key idea:
    - Estimate a noise-level map from the noisy image.
    - Condition a non-blind denoiser on that estimated map.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        est_width: int = 24,
        est_depth: int = 5,
        den_width: int = 48,
        den_depth: int = 2,
    ) -> None:
        super().__init__()
        c = int(in_channels)
        if c <= 0:
            raise ValueError("in_channels must be > 0")

        self.est = NoiseEstimationNet(in_channels=c, width=int(est_width), depth=int(est_depth))
        # conditional input: (noisy, noise_map) -> channels = 2C
        self.denoiser = ConditionalDenoiserUNet(in_channels=c * 2, width=int(den_width), depth=int(den_depth))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        if x.ndim != 4:
            raise ValueError(f"Expected input shape (B, C, H, W), got {tuple(x.shape)}")

        x_pad, pad_hw = pad_to_multiple(x, 4, mode="reflect")
        noise_map = self.est(x_pad)
        cond = torch.cat([x_pad, noise_map], dim=1)
        residual = self.denoiser(cond)
        y = x_pad - residual
        return unpad(y, pad_hw)


_VARIANTS: dict[str, dict] = {
    "cbdnet_tiny": {"est_w": 16, "est_d": 4, "den_w": 32, "den_d": 1},
    "cbdnet_small": {"est_w": 24, "est_d": 5, "den_w": 48, "den_d": 2},
    "cbdnet_base": {"est_w": 32, "est_d": 6, "den_w": 64, "den_d": 2},
}


def build_cbdnet_denoiser(
    *,
    in_channels: int,
    variant: str = "cbdnet_small",
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown CBDNet variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    return CBDNet(
        in_channels=int(in_channels),
        est_width=int(spec["est_w"]),
        est_depth=int(spec["est_d"]),
        den_width=int(spec["den_w"]),
        den_depth=int(spec["den_d"]),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.rand(2, 1, 64, 64)
    noisy = (x + torch.randn_like(x) * 0.15).clamp(0.0, 1.0)
    m = build_cbdnet_denoiser(in_channels=1, variant="cbdnet_tiny")
    y = m(noisy)
    print("cbdnet_tiny", tuple(y.shape))
    loss = (y - x).pow(2).mean()
    loss.backward()
    print("ok")

