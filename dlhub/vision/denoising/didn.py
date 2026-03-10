import torch
import torch.nn.functional as F
from torch import nn

from ._utils import pad_to_multiple, unpad


class DenseLayer(nn.Module):
    def __init__(self, in_ch: int, growth: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(int(in_ch), int(growth), kernel_size=3, padding=1, bias=True)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act(self.conv(x))
        return torch.cat([x, y], dim=1)


class DenseBlock(nn.Module):
    def __init__(self, in_ch: int, *, num_layers: int, growth: int, out_ch: int) -> None:
        super().__init__()
        c_in = int(in_ch)
        n = int(num_layers)
        g = int(growth)
        c_out = int(out_ch)
        if n <= 0:
            raise ValueError("num_layers must be > 0")
        if g <= 0:
            raise ValueError("growth must be > 0")
        if c_out <= 0:
            raise ValueError("out_ch must be > 0")

        layers: list[nn.Module] = []
        ch = c_in
        for _ in range(n):
            layers.append(DenseLayer(ch, g))
            ch += g
        self.layers = nn.Sequential(*layers)
        self.compress = nn.Conv2d(ch, c_out, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.layers(x)
        y = self.compress(y)
        return F.relu(y + x, inplace=True) if y.shape == x.shape else F.relu(y, inplace=True)


class DIDN(nn.Module):
    """DIDN-style denoiser (toy-first, pure torch).

    DIDN (Densely connected Iterative Down-Up Network) is commonly used as a blind denoiser.
    This is a simplified two-level down/up version with dense blocks.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        width: int = 32,
        num_blocks: int = 3,
        layers_per_block: int = 4,
        growth: int = 12,
    ) -> None:
        super().__init__()
        c_in = int(in_channels)
        w0 = int(width)
        nb = int(num_blocks)
        lp = int(layers_per_block)
        gr = int(growth)
        if c_in <= 0:
            raise ValueError("in_channels must be > 0")
        if w0 < 8:
            raise ValueError("width must be >= 8")
        if nb <= 0:
            raise ValueError("num_blocks must be > 0")

        self.intro = nn.Conv2d(c_in, w0, kernel_size=3, padding=1, bias=True)

        self.down1 = nn.Conv2d(w0, w0 * 2, kernel_size=3, stride=2, padding=1, bias=True)
        self.down2 = nn.Conv2d(w0 * 2, w0 * 4, kernel_size=3, stride=2, padding=1, bias=True)

        self.blocks_l2 = nn.Sequential(
            *[DenseBlock(w0 * 4, num_layers=lp, growth=gr, out_ch=w0 * 4) for _ in range(nb)]
        )

        self.up2 = nn.Conv2d(w0 * 4, w0 * 2, kernel_size=1, bias=True)
        self.merge1 = nn.Conv2d(w0 * 4, w0 * 2, kernel_size=1, bias=True)
        self.blocks_l1 = nn.Sequential(
            *[
                DenseBlock(
                    w0 * 2, num_layers=max(1, lp // 2), growth=max(4, gr // 2), out_ch=w0 * 2
                )
                for _ in range(max(1, nb - 1))
            ]
        )

        self.up1 = nn.Conv2d(w0 * 2, w0, kernel_size=1, bias=True)
        self.merge0 = nn.Conv2d(w0 * 2, w0, kernel_size=1, bias=True)

        self.outro = nn.Conv2d(w0, c_in, kernel_size=3, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        if x.ndim != 4:
            raise ValueError(f"Expected input shape (B, C, H, W), got {tuple(x.shape)}")

        x_pad, pad_hw = pad_to_multiple(x, 4, mode="reflect")
        inp = x_pad

        f0 = F.relu(self.intro(x_pad), inplace=True)
        f1 = F.relu(self.down1(f0), inplace=True)
        f2 = F.relu(self.down2(f1), inplace=True)

        f2 = self.blocks_l2(f2)

        u1 = F.interpolate(f2, scale_factor=2, mode="nearest")
        u1 = F.relu(self.up2(u1), inplace=True)
        u1 = self.blocks_l1(F.relu(self.merge1(torch.cat([u1, f1], dim=1)), inplace=True))

        u0 = F.interpolate(u1, scale_factor=2, mode="nearest")
        u0 = F.relu(self.up1(u0), inplace=True)
        u0 = F.relu(self.merge0(torch.cat([u0, f0], dim=1)), inplace=True)

        residual = self.outro(u0)
        y = inp - residual
        return unpad(y, pad_hw)


_VARIANTS: dict[str, dict] = {
    "didn_tiny": {"width": 24, "blocks": 2, "layers": 3, "growth": 8},
    "didn_small": {"width": 32, "blocks": 3, "layers": 4, "growth": 12},
    "didn_base": {"width": 48, "blocks": 4, "layers": 5, "growth": 16},
}


def build_didn_denoiser(
    *,
    in_channels: int,
    variant: str = "didn_small",
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown DIDN variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    return DIDN(
        in_channels=int(in_channels),
        width=int(spec["width"]),
        num_blocks=int(spec["blocks"]),
        layers_per_block=int(spec["layers"]),
        growth=int(spec["growth"]),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.rand(2, 1, 64, 64)
    noisy = (x + torch.randn_like(x) * 0.1).clamp(0.0, 1.0)
    m = build_didn_denoiser(in_channels=1, variant="didn_tiny")
    y = m(noisy)
    print("didn_tiny", tuple(y.shape))
    loss = (y - x).pow(2).mean()
    loss.backward()
    print("ok")
