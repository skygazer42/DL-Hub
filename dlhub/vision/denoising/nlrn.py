import torch
import torch.nn.functional as F
from torch import nn


class _ResBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        c = int(channels)
        self.conv1 = nn.Conv2d(c, c, kernel_size=3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(c, c, kernel_size=3, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.conv1(x), inplace=True)
        y = self.conv2(y)
        return F.relu(x + y, inplace=True)


class NonLocalBlock(nn.Module):
    """Non-local block (self-attention over spatial positions), toy-first."""

    def __init__(self, channels: int, *, inter_channels: int | None = None) -> None:
        super().__init__()
        c = int(channels)
        ic = int(inter_channels) if inter_channels is not None else max(8, c // 2)
        if c <= 0:
            raise ValueError("channels must be > 0")
        if ic <= 0:
            raise ValueError("inter_channels must be > 0")

        self.theta = nn.Conv2d(c, ic, kernel_size=1, bias=True)
        self.phi = nn.Conv2d(c, ic, kernel_size=1, bias=True)
        self.g = nn.Conv2d(c, ic, kernel_size=1, bias=True)
        self.out = nn.Conv2d(ic, c, kernel_size=1, bias=True)

        # Start as identity for stability (common in non-local blocks).
        self.gamma = nn.Parameter(torch.zeros(()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        hw = h * w

        theta = self.theta(x).view(b, -1, hw).transpose(1, 2)  # (B, HW, C')
        phi = self.phi(x).view(b, -1, hw)  # (B, C', HW)
        attn = torch.matmul(theta, phi)  # (B, HW, HW)
        attn = torch.softmax(attn, dim=-1)

        g = self.g(x).view(b, -1, hw).transpose(1, 2)  # (B, HW, C')
        y = torch.matmul(attn, g)  # (B, HW, C')
        y = y.transpose(1, 2).contiguous().view(b, -1, h, w)
        y = self.out(y)
        return x + y * self.gamma


class NLRN(nn.Module):
    """NLRN (Non-Local Recurrent Network) style denoiser (toy-first).

    Applies a recurrent unit composed of residual conv + non-local block several times.
    Predicts a residual/noise map and returns `x - residual`.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        features: int = 64,
        recurrences: int = 4,
    ) -> None:
        super().__init__()
        c_in = int(in_channels)
        f = int(features)
        r = int(recurrences)
        if c_in <= 0:
            raise ValueError("in_channels must be > 0")
        if f < 8:
            raise ValueError("features must be >= 8")
        if r <= 0:
            raise ValueError("recurrences must be > 0")

        self.in_conv = nn.Conv2d(c_in, f, kernel_size=3, padding=1, bias=True)
        self.res = _ResBlock(f)
        self.nl = NonLocalBlock(f)
        self.out_conv = nn.Conv2d(f, c_in, kernel_size=3, padding=1, bias=True)
        self.recurrences = r

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        if x.ndim != 4:
            raise ValueError(f"Expected input shape (B, C, H, W), got {tuple(x.shape)}")

        y = F.relu(self.in_conv(x), inplace=True)
        for _ in range(self.recurrences):
            y = self.res(y)
            y = self.nl(y)
        residual = self.out_conv(y)
        return x - residual


_VARIANTS: dict[str, dict] = {
    "nlrn_tiny": {"features": 32, "recurrences": 2},
    "nlrn_small": {"features": 48, "recurrences": 4},
    "nlrn_base": {"features": 64, "recurrences": 6},
}


def build_nlrn_denoiser(*, in_channels: int, variant: str = "nlrn_small") -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown NLRN variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    return NLRN(
        in_channels=int(in_channels),
        features=int(spec["features"]),
        recurrences=int(spec["recurrences"]),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.rand(2, 1, 32, 32)
    noisy = (x + torch.randn_like(x) * 0.1).clamp(0.0, 1.0)
    m = build_nlrn_denoiser(in_channels=1, variant="nlrn_tiny")
    y = m(noisy)
    print("nlrn_tiny", tuple(y.shape))
    loss = (y - x).pow(2).mean()
    loss.backward()
    print("ok")
