
import torch
from torch import nn
import torch.nn.functional as F


def _smooth_h(x: torch.Tensor, *, k: int, padding: str) -> torch.Tensor:
    if x.ndim != 4:
        raise ValueError(f"Expected NCHW, got {tuple(x.shape)}")
    k = int(k)
    if k < 1 or k % 2 == 0:
        raise ValueError("k must be odd and >= 1")
    if k == 1:
        return x
    p = k // 2
    _, c, _, _ = x.shape
    weight = torch.ones((c, 1, k, 1), device=x.device, dtype=x.dtype) / float(k)
    x_pad = F.pad(x, (0, 0, p, p), mode=str(padding))
    return F.conv2d(x_pad, weight, bias=None, stride=1, padding=0, groups=c)


def _smooth_w(x: torch.Tensor, *, k: int, padding: str) -> torch.Tensor:
    if x.ndim != 4:
        raise ValueError(f"Expected NCHW, got {tuple(x.shape)}")
    k = int(k)
    if k < 1 or k % 2 == 0:
        raise ValueError("k must be odd and >= 1")
    if k == 1:
        return x
    p = k // 2
    _, c, _, _ = x.shape
    weight = torch.ones((c, 1, 1, k), device=x.device, dtype=x.dtype) / float(k)
    x_pad = F.pad(x, (p, p, 0, 0), mode=str(padding))
    return F.conv2d(x_pad, weight, bias=None, stride=1, padding=0, groups=c)


class RowColBiasCorrector(nn.Module):
    """Remove row/col fixed-pattern bias by estimating per-row/col mean offsets.

    Matches `noise_type=rowcol_bias` in Lesson 10.
    """

    def __init__(
        self,
        *,
        smooth: int = 7,
        strength: float = 1.0,
        padding: str = "reflect",
        clamp: bool = True,
    ) -> None:
        super().__init__()
        k = int(smooth)
        if k < 1 or k % 2 == 0:
            raise ValueError("smooth must be odd and >= 1")
        s = float(strength)
        if s < 0.0:
            raise ValueError("strength must be >= 0")
        self.smooth = k
        self.strength = s
        self.padding = str(padding)
        self.clamp = bool(clamp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        if x.ndim != 4:
            raise ValueError(f"Expected input shape (B, C, H, W), got {tuple(x.shape)}")

        if float(self.strength) == 0.0:
            return x.clamp(0.0, 1.0) if self.clamp else x

        g = x.mean(dim=(-2, -1), keepdim=True)  # (B,C,1,1)
        row_mean = x.mean(dim=-1, keepdim=True)  # (B,C,H,1)
        col_mean = x.mean(dim=-2, keepdim=True)  # (B,C,1,W)

        row_bias = row_mean - g
        col_bias = col_mean - g

        if int(self.smooth) > 1:
            row_bias = _smooth_h(row_bias, k=int(self.smooth), padding=self.padding)
            col_bias = _smooth_w(col_bias, k=int(self.smooth), padding=self.padding)

        bias = row_bias + col_bias
        y = x - float(self.strength) * bias
        return y.clamp(0.0, 1.0) if self.clamp else y


_VARIANTS: dict[str, dict] = {
    "rowcol_bias_tiny": {"smooth": 1},
    "rowcol_bias_small": {"smooth": 7},
    "rowcol_bias_base": {"smooth": 15},
}


def build_rowcol_bias_corrector_denoiser(
    *,
    in_channels: int,  # unused
    sigma: float = 0.1,
    variant: str = "rowcol_bias_small",
) -> nn.Module:
    _ = int(in_channels)
    _ = float(sigma)
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown RowColBiasCorrector variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    return RowColBiasCorrector(smooth=int(spec["smooth"]), strength=1.0, padding="reflect", clamp=True)


if __name__ == "__main__":
    torch.manual_seed(0)
    clean = torch.zeros(1, 1, 32, 32)
    clean[:, :, 10:22, 12:20] = 1.0
    # Add synthetic row/col bias.
    row = torch.randn(1, 1, 32, 1) * 0.03
    col = torch.randn(1, 1, 1, 32) * 0.03
    noisy = (clean + row + col).clamp(0.0, 1.0)
    m = build_rowcol_bias_corrector_denoiser(in_channels=1, variant="rowcol_bias_tiny")
    out = m(noisy)
    print("rowcol_bias_tiny", tuple(out.shape), float((out - clean).pow(2).mean().item()))

