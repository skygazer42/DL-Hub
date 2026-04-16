from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    hidden_features: int = 64
    image_size: int = 24
    min_sigma: float = 0.02


def render_gaussians(
    centers: torch.Tensor, sigmas: torch.Tensor, amplitudes: torch.Tensor, *, image_size: int
) -> torch.Tensor:
    if centers.ndim != 3 or centers.size(-1) != 2:
        raise ValueError("centers must be shaped [batch, num_points, 2]")
    if sigmas.shape != centers.shape[:2]:
        raise ValueError("sigmas must be shaped [batch, num_points]")
    if amplitudes.shape != centers.shape[:2]:
        raise ValueError("amplitudes must be shaped [batch, num_points]")

    device = centers.device
    dtype = centers.dtype
    grid = torch.linspace(-1.0, 1.0, int(image_size), device=device, dtype=dtype)
    yy, xx = torch.meshgrid(grid, grid, indexing="ij")
    xx = xx.view(1, 1, int(image_size), int(image_size))
    yy = yy.view(1, 1, int(image_size), int(image_size))

    cx = centers[..., 0].unsqueeze(-1).unsqueeze(-1)
    cy = centers[..., 1].unsqueeze(-1).unsqueeze(-1)
    var = sigmas.square().unsqueeze(-1).unsqueeze(-1).clamp(min=1e-6)
    sq_dist = (xx - cx).square() + (yy - cy).square()
    gaussians = torch.exp(-0.5 * sq_dist / var) * amplitudes.unsqueeze(-1).unsqueeze(-1)
    return gaussians.mean(dim=1, keepdim=True)


class ToyGaussianSplattingModel(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        hidden = int(cfg.hidden_features)
        if hidden < 4:
            raise ValueError("hidden_features must be >= 4")
        if int(cfg.image_size) < 8:
            raise ValueError("image_size must be >= 8")
        if float(cfg.min_sigma) <= 0.0:
            raise ValueError("min_sigma must be > 0")

        self.cfg = cfg
        self.point_head = nn.Sequential(
            nn.Linear(3, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 4),
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        if points.ndim != 3 or points.size(-1) != 3:
            raise ValueError("expected points shaped [batch, num_points, 3]")

        batch_size, num_points, channels = points.shape
        params = self.point_head(points.reshape(batch_size * num_points, channels))
        params = params.reshape(batch_size, num_points, 4)

        delta_xy = torch.tanh(params[..., :2]) * 0.25
        centers = (points[..., :2] + delta_xy).clamp(-1.0, 1.0)
        sigmas = F.softplus(params[..., 2]) + float(self.cfg.min_sigma)
        amplitudes = F.softplus(params[..., 3]) + 1e-4
        return render_gaussians(
            centers=centers,
            sigmas=sigmas,
            amplitudes=amplitudes,
            image_size=int(self.cfg.image_size),
        )


def gaussian_splatting_loss(
    pred_image: torch.Tensor, target_image: torch.Tensor
) -> tuple[torch.Tensor, dict[str, float]]:
    if pred_image.shape != target_image.shape:
        raise ValueError("pred_image and target_image must have the same shape")

    mse = (pred_image - target_image).square().mean()
    mass_l1 = (pred_image.mean(dim=(-2, -1)) - target_image.mean(dim=(-2, -1))).abs().mean()
    total = mse + 0.1 * mass_l1
    return total, {"mse": float(mse.detach().item()), "mass_l1": float(mass_l1.detach().item())}


__all__ = ["ModelConfig", "ToyGaussianSplattingModel", "gaussian_splatting_loss", "render_gaussians"]
