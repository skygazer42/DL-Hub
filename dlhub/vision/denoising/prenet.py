
"""PReNet (Progressive Recurrent Network) - toy-first implementation.

Reference (original idea):
- "Progressive Image Deraining Networks: A Better and Simpler Baseline" (CVPR 2019)

Toy interpretation:
- Use a ConvGRU-style hidden state updated over multiple stages.
- Each stage predicts a rain residual to subtract from the current estimate.
"""

import torch
from torch import nn


class ConvGRUCell(nn.Module):
    def __init__(self, *, in_channels: int, hidden_channels: int) -> None:
        super().__init__()
        c_in = int(in_channels)
        c_h = int(hidden_channels)
        if c_in <= 0:
            raise ValueError("in_channels must be > 0")
        if c_h <= 0:
            raise ValueError("hidden_channels must be > 0")

        self.conv_z = nn.Conv2d(c_in + c_h, c_h, kernel_size=3, padding=1, bias=True)
        self.conv_r = nn.Conv2d(c_in + c_h, c_h, kernel_size=3, padding=1, bias=True)
        self.conv_h = nn.Conv2d(c_in + c_h, c_h, kernel_size=3, padding=1, bias=True)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        # x: (B, C_in, H, W), h: (B, C_h, H, W)
        if h is None:
            raise ValueError("Hidden state h must be a Tensor (pass zeros for init).")
        combined = torch.cat([x, h], dim=1)
        z = torch.sigmoid(self.conv_z(combined))
        r = torch.sigmoid(self.conv_r(combined))
        combined_r = torch.cat([x, r * h], dim=1)
        h_tilde = torch.tanh(self.conv_h(combined_r))
        return (1.0 - z) * h + z * h_tilde


class ResBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        c = int(channels)
        self.conv1 = nn.Conv2d(c, c, kernel_size=3, padding=1, bias=True)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(c, c, kernel_size=3, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv2(self.act(self.conv1(x)))
        return x + y


class PReNet(nn.Module):
    """Toy PReNet-style derainer (progressive recurrent refinement)."""

    def __init__(
        self,
        *,
        in_channels: int,
        hidden: int = 24,
        stages: int = 6,
        depth: int = 1,
    ) -> None:
        super().__init__()
        c_in = int(in_channels)
        h = int(hidden)
        t = int(stages)
        d = int(depth)
        if c_in <= 0:
            raise ValueError("in_channels must be > 0")
        if h < 8:
            raise ValueError("hidden must be >= 8")
        if t <= 0:
            raise ValueError("stages must be > 0")
        if d <= 0:
            raise ValueError("depth must be > 0")

        self.cell = ConvGRUCell(in_channels=c_in, hidden_channels=h)
        self.refine = nn.Sequential(*[ResBlock(h) for _ in range(d)])
        self.head = nn.Conv2d(h, c_in, kernel_size=3, padding=1, bias=True)
        self.stages = t
        self.hidden = h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        if x.ndim != 4:
            raise ValueError(f"Expected input shape (B, C, H, W), got {tuple(x.shape)}")

        y = x
        h = torch.zeros((x.shape[0], int(self.hidden), x.shape[2], x.shape[3]), device=x.device, dtype=x.dtype)
        for _ in range(int(self.stages)):
            h = self.cell(y, h)
            feat = self.refine(h)
            residual = self.head(feat)
            y = y - residual
        return y


_VARIANTS: dict[str, dict] = {
    "prenet_tiny": {"hidden": 16, "stages": 4, "depth": 1},
    "prenet_small": {"hidden": 24, "stages": 6, "depth": 1},
    "prenet_base": {"hidden": 32, "stages": 8, "depth": 2},
}


def build_prenet_denoiser(
    *,
    in_channels: int,
    variant: str = "prenet_small",
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown PReNet variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    return PReNet(
        in_channels=int(in_channels),
        hidden=int(spec["hidden"]),
        stages=int(spec["stages"]),
        depth=int(spec["depth"]),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.rand(2, 1, 64, 64)
    noisy = (x + torch.randn_like(x) * 0.1).clamp(0.0, 1.0)
    m = build_prenet_denoiser(in_channels=1, variant="prenet_tiny")
    y = m(noisy)
    print("prenet_tiny", tuple(y.shape))
    loss = (y - x).pow(2).mean()
    loss.backward()
    print("ok")

