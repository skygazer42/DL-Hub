import torch
import torch.nn.functional as F
from torch import nn


def _smooth_h(x: torch.Tensor, *, k: int, padding: str) -> torch.Tensor:
    """Per-channel 1D box smoothing along H for NCHW tensors (implemented via depthwise conv2d)."""

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
    """Per-channel 1D box smoothing along W for NCHW tensors (implemented via depthwise conv2d)."""

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


class StripeRemover(nn.Module):
    """A simple stripe remover for sinusoidal banding (torch-only, compact-first).

    Idea:
    - Stripe noise is often approximately constant along one axis.
    - Estimate a 1D stripe profile by averaging across the orthogonal axis.
    - Smooth the profile to reduce content leakage, center it, then subtract.
    - Auto-select vertical vs horizontal stripes by comparing profile energy.
    """

    def __init__(
        self,
        *,
        smooth: int = 9,
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

        # Horizontal profile: varies along H, constant across W.
        row_prof = x.mean(dim=-1, keepdim=True)  # (B,C,H,1)
        # Vertical profile: varies along W, constant across H.
        col_prof = x.mean(dim=-2, keepdim=True)  # (B,C,1,W)

        row_prof_s = _smooth_h(row_prof, k=int(self.smooth), padding=self.padding)
        col_prof_s = _smooth_w(col_prof, k=int(self.smooth), padding=self.padding)

        # Center (remove DC).
        row_prof_s = row_prof_s - row_prof_s.mean(dim=-2, keepdim=True)
        col_prof_s = col_prof_s - col_prof_s.mean(dim=-1, keepdim=True)

        row_energy = row_prof_s.pow(2).mean()
        col_energy = col_prof_s.pow(2).mean()

        stripe = torch.where(
            (row_energy >= col_energy),
            row_prof_s.expand_as(x),
            col_prof_s.expand_as(x),
        )

        y = x - float(self.strength) * stripe
        return y.clamp(0.0, 1.0) if self.clamp else y


_VARIANTS: dict[str, dict] = {
    "stripe_remover_tiny": {"smooth": 7, "strength": 1.0},
    "stripe_remover_small": {"smooth": 11, "strength": 1.0},
    "stripe_remover_base": {"smooth": 15, "strength": 1.0},
}


def build_stripe_remover_denoiser(
    *,
    in_channels: int,  # unused (kept for consistent signatures)
    sigma: float = 0.1,  # unused (kept for consistent signatures)
    variant: str = "stripe_remover_small",
) -> nn.Module:
    _ = int(in_channels)
    _ = float(sigma)
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(
            f"Unknown StripeRemover variant: {variant!r}. Supported: {sorted(_VARIANTS)}"
        )
    spec = _VARIANTS[name]
    return StripeRemover(
        smooth=int(spec["smooth"]),
        strength=float(spec["strength"]),
        padding="reflect",
        clamp=True,
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    clean = torch.rand(1, 1, 64, 64)
    # Add synthetic vertical stripe
    w = clean.shape[-1]
    coord = torch.arange(w, dtype=torch.float32)[None, None, None, :]
    stripe = 0.12 * torch.sin(2.0 * 3.141592653589793 * coord / 8.0)
    noisy = (clean + stripe).clamp(0.0, 1.0)

    m = build_stripe_remover_denoiser(in_channels=1, variant="stripe_remover_tiny")
    out = m(noisy)
    print("stripe_remover_tiny", tuple(out.shape), float((out - clean).pow(2).mean().item()))
