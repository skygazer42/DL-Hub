import torch
import torch.nn.functional as F
from torch import nn


def _pad_replicate(x: torch.Tensor) -> torch.Tensor:
    return F.pad(x, (1, 1, 1, 1), mode="replicate")


class AnisotropicDiffusion(nn.Module):
    """Perona–Malik anisotropic diffusion (torch-only, compact-first).

    Iterative PDE-style denoiser:
      u_{t+1} = u_t + step * sum_dir c(|∇u|) * ∇u

    Conduction functions:
    - "exp":  c(s) = exp(-(s / kappa)^2)
    - "frac": c(s) = 1 / (1 + (s / kappa)^2)
    """

    def __init__(
        self,
        *,
        n_iter: int = 12,
        kappa: float = 0.15,
        step: float = 0.2,
        conduction: str = "exp",
        clamp: bool = True,
    ) -> None:
        super().__init__()
        it = int(n_iter)
        if it <= 0:
            raise ValueError("n_iter must be > 0")
        k = float(kappa)
        if k <= 0.0:
            raise ValueError("kappa must be > 0")
        s = float(step)
        if not (0.0 < s <= 0.25):
            raise ValueError("step should be in (0, 0.25] for stability")
        conduction = str(conduction).lower().strip()
        if conduction not in {"exp", "frac"}:
            raise ValueError("conduction must be 'exp' or 'frac'")

        self.n_iter = it
        self.kappa = k
        self.step = s
        self.conduction = conduction
        self.clamp = bool(clamp)

    def _c(self, g: torch.Tensor) -> torch.Tensor:
        z = g / float(self.kappa)
        if self.conduction == "exp":
            return torch.exp(-(z * z))
        return 1.0 / (1.0 + z * z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        if x.ndim != 4:
            raise ValueError(f"Expected input shape (B, C, H, W), got {tuple(x.shape)}")

        u = x
        step = float(self.step)
        for _ in range(int(self.n_iter)):
            up = _pad_replicate(u)
            c0 = up[:, :, 1:-1, 1:-1]
            n = up[:, :, :-2, 1:-1] - c0
            s = up[:, :, 2:, 1:-1] - c0
            w = up[:, :, 1:-1, :-2] - c0
            e = up[:, :, 1:-1, 2:] - c0

            c_n = self._c(n.abs())
            c_s = self._c(s.abs())
            c_w = self._c(w.abs())
            c_e = self._c(e.abs())

            u = u + step * (c_n * n + c_s * s + c_w * w + c_e * e)

        return u.clamp(0.0, 1.0) if self.clamp else u


_VARIANTS: dict[str, dict] = {
    "anisodiff_fast": {"iters": 6, "kappa": 0.18, "step": 0.22, "cond": "exp"},
    "anisodiff_quality": {"iters": 14, "kappa": 0.15, "step": 0.18, "cond": "exp"},
    "anisodiff_frac": {"iters": 12, "kappa": 0.15, "step": 0.18, "cond": "frac"},
}


def build_anisotropic_diffusion_denoiser(
    *,
    in_channels: int,  # unused (kept for consistent signatures)
    sigma: float = 0.1,
    variant: str = "anisodiff_quality",
) -> nn.Module:
    _ = int(in_channels)
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(
            f"Unknown AnisotropicDiffusion variant: {variant!r}. Supported: {sorted(_VARIANTS)}"
        )
    spec = _VARIANTS[name]

    # Heuristic: for larger sigma, reduce kappa a bit (stronger edge stopping).
    kappa = float(spec["kappa"]) / (1.0 + 0.5 * float(sigma))
    return AnisotropicDiffusion(
        n_iter=int(spec["iters"]),
        kappa=max(1e-4, kappa),
        step=float(spec["step"]),
        conduction=str(spec["cond"]),
        clamp=True,
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.rand(1, 1, 64, 64)
    noisy = (x + torch.randn_like(x) * 0.12).clamp(0.0, 1.0)
    m = build_anisotropic_diffusion_denoiser(in_channels=1, sigma=0.12, variant="anisodiff_fast")
    y = m(noisy)
    print("anisodiff_fast", tuple(y.shape), float((y - x).pow(2).mean().item()))
