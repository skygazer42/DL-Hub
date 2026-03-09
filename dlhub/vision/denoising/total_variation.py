
import torch
from torch import nn


def _grad_forward(u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward differences (dx, dy) for NCHW tensor."""

    dx = torch.zeros_like(u)
    dy = torch.zeros_like(u)
    dx[..., :, :-1] = u[..., :, 1:] - u[..., :, :-1]
    dy[..., :-1, :] = u[..., 1:, :] - u[..., :-1, :]
    return dx, dy


def _div_backward(px: torch.Tensor, py: torch.Tensor) -> torch.Tensor:
    """Divergence of a vector field (px, py) for NCHW tensors (backward differences)."""

    div = torch.zeros_like(px)
    # x-component
    div[..., :, 0] = px[..., :, 0]
    div[..., :, 1:-1] = px[..., :, 1:-1] - px[..., :, :-2]
    div[..., :, -1] = -px[..., :, -2]
    # y-component
    div[..., 0, :] += py[..., 0, :]
    div[..., 1:-1, :] += py[..., 1:-1, :] - py[..., :-2, :]
    div[..., -1, :] += -py[..., -2, :]
    return div


class TotalVariationDenoiser(nn.Module):
    """ROF / TV denoising via Chambolle iterations (torch-only, toy-first).

    Solves (approximately):
        min_u 0.5 * ||u - f||^2 + weight * TV(u)

    Notes:
    - This is intended as a classical baseline and is not optimized for speed.
    - For toy images (32-128px) it's fine on CPU.
    """

    def __init__(
        self,
        *,
        weight: float = 0.1,
        n_iter: int = 20,
        tau: float = 0.25,
        clamp: bool = True,
    ) -> None:
        super().__init__()
        w = float(weight)
        if w <= 0.0:
            raise ValueError("weight must be > 0")
        it = int(n_iter)
        if it <= 0:
            raise ValueError("n_iter must be > 0")
        t = float(tau)
        if not (0.0 < t <= 0.25):
            raise ValueError("tau should be in (0, 0.25] for stability")

        self.weight = w
        self.n_iter = it
        self.tau = t
        self.clamp = bool(clamp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        if x.ndim != 4:
            raise ValueError(f"Expected input shape (B, C, H, W), got {tuple(x.shape)}")

        f = x
        px = torch.zeros_like(f)
        py = torch.zeros_like(f)

        inv_w = 1.0 / float(self.weight)
        tau = float(self.tau)

        for _ in range(int(self.n_iter)):
            div_p = _div_backward(px, py)
            u = f - float(self.weight) * div_p
            ux, uy = _grad_forward(u)

            px_new = px + (tau * inv_w) * ux
            py_new = py + (tau * inv_w) * uy
            norm = torch.sqrt(px_new * px_new + py_new * py_new).clamp_min(1.0)
            px = px_new / norm
            py = py_new / norm

        out = f - float(self.weight) * _div_backward(px, py)
        return out.clamp(0.0, 1.0) if self.clamp else out


_VARIANTS: dict[str, dict] = {
    "tv_fast": {"iters": 10, "weight": 0.08},
    "tv_quality": {"iters": 30, "weight": 0.12},
    "tv_strong": {"iters": 50, "weight": 0.18},
}


def build_total_variation_denoiser(
    *,
    in_channels: int,  # unused (kept for consistent signatures)
    sigma: float = 0.1,
    variant: str = "tv_quality",
) -> nn.Module:
    _ = int(in_channels)
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown TotalVariation variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]

    # Heuristic: increase weight slightly with sigma (more noise -> more smoothing).
    weight = float(spec["weight"]) * (1.0 + 0.75 * float(sigma))
    return TotalVariationDenoiser(weight=weight, n_iter=int(spec["iters"]), tau=0.25, clamp=True)


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.rand(1, 1, 64, 64)
    noisy = (x + torch.randn_like(x) * 0.12).clamp(0.0, 1.0)
    m = build_total_variation_denoiser(in_channels=1, sigma=0.12, variant="tv_fast")
    y = m(noisy)
    print("tv_fast", tuple(y.shape), float((y - x).pow(2).mean().item()))

