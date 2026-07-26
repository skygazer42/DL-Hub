from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    hidden_dim: int = 128
    time_embed_dim: int = 32


@dataclass(frozen=True)
class ConsistencySchedule:
    """Karras-style discretized noise grid used for consistency training.

    `sigmas()` returns `num_steps` noise levels increasing from `sigma_min`
    to `sigma_max`. The consistency function is anchored at `sigma_min`,
    where it must be the identity.
    """

    num_steps: int = 20
    sigma_min: float = 0.02
    sigma_max: float = 3.0
    rho: float = 7.0
    sigma_data: float = 0.5

    def sigmas(self) -> torch.Tensor:
        steps = torch.arange(int(self.num_steps), dtype=torch.float32) / max(1, int(self.num_steps) - 1)
        inv_rho = 1.0 / float(self.rho)
        lo = float(self.sigma_min) ** inv_rho
        hi = float(self.sigma_max) ** inv_rho
        return (lo + steps * (hi - lo)) ** float(self.rho)


def sigma_embedding(sigmas: torch.Tensor, dim: int) -> torch.Tensor:
    half_dim = max(1, dim // 2)
    device = sigmas.device
    exponent = torch.arange(half_dim, device=device, dtype=torch.float32) / float(half_dim)
    frequencies = torch.exp(-torch.log(torch.tensor(10000.0, device=device)) * exponent)
    angles = torch.log(sigmas.float().clamp_min(1e-8)).unsqueeze(1) * frequencies.unsqueeze(0)
    emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
    if emb.shape[1] < dim:
        emb = torch.cat([emb, torch.zeros((emb.shape[0], dim - emb.shape[1]), device=device)], dim=1)
    return emb[:, :dim]


class ConsistencyModel(nn.Module):
    """A tiny consistency function over flattened 28x28 images.

    Implements the skip parameterization from Song et al. (2023):
    ``f(x, sigma) = c_skip(sigma) * x + c_out(sigma) * F(x, sigma)`` with
    ``c_skip(sigma_min) = 1`` and ``c_out(sigma_min) = 0``, so the boundary
    condition ``f(x, sigma_min) = x`` holds by construction.
    """

    def __init__(self, cfg: ModelConfig, schedule: ConsistencySchedule | None = None) -> None:
        super().__init__()
        self.cfg = cfg
        self.schedule = schedule if schedule is not None else ConsistencySchedule()
        in_dim = 28 * 28
        self.net = nn.Sequential(
            nn.Linear(in_dim + cfg.time_embed_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, in_dim),
        )

    def _coefficients(self, sigmas: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sigma_min = float(self.schedule.sigma_min)
        sigma_data = float(self.schedule.sigma_data)
        shifted = sigmas - sigma_min
        c_skip = sigma_data**2 / (shifted**2 + sigma_data**2)
        c_out = sigma_data * shifted / torch.sqrt(sigmas**2 + sigma_data**2)
        return c_skip.unsqueeze(1), c_out.unsqueeze(1)

    def forward(self, xt: torch.Tensor, sigmas: torch.Tensor) -> torch.Tensor:
        if sigmas.ndim != 1:
            raise ValueError(f"Expected `sigmas` shape (B,), got {tuple(sigmas.shape)}")
        sigma_emb = sigma_embedding(sigmas, self.cfg.time_embed_dim).to(dtype=xt.dtype)
        residual = self.net(torch.cat([xt, sigma_emb], dim=1))
        c_skip, c_out = self._coefficients(sigmas.to(dtype=xt.dtype))
        return c_skip * xt + c_out * residual


def consistency_training_loss(
    model: ConsistencyModel,
    target_model: ConsistencyModel,
    x0: torch.Tensor,
    schedule: ConsistencySchedule,
) -> torch.Tensor:
    """Self-consistency loss between adjacent noise levels sharing one noise draw."""
    sigmas = schedule.sigmas().to(device=x0.device, dtype=x0.dtype)
    indices = torch.randint(0, int(schedule.num_steps) - 1, (x0.size(0),), device=x0.device)
    sigma_curr = sigmas.index_select(0, indices)
    sigma_next = sigmas.index_select(0, indices + 1)

    noise = torch.randn_like(x0)
    x_next = x0 + sigma_next.unsqueeze(1) * noise
    x_curr = x0 + sigma_curr.unsqueeze(1) * noise

    pred = model(x_next, sigma_next)
    with torch.no_grad():
        target = target_model(x_curr, sigma_curr)
    return torch.nn.functional.mse_loss(pred, target)


@torch.no_grad()
def update_ema(target_model: ConsistencyModel, model: ConsistencyModel, decay: float) -> None:
    for target_param, param in zip(target_model.parameters(), model.parameters()):
        target_param.mul_(decay).add_(param, alpha=1.0 - decay)


@torch.no_grad()
def sample_consistency(
    model: ConsistencyModel,
    schedule: ConsistencySchedule,
    *,
    num_samples: int,
    device: torch.device,
    num_steps: int = 1,
    return_all: bool = False,
) -> torch.Tensor:
    """One-step generation, with optional multistep stochastic refinement."""
    step_count = max(1, int(num_steps))
    sigma_min = float(schedule.sigma_min)
    sigma_max = float(schedule.sigma_max)

    xt = sigma_max * torch.randn((int(num_samples), 28 * 28), device=device)
    x = model(xt, torch.full((int(num_samples),), sigma_max, device=device))
    frames = [x.view(-1, 1, 28, 28).clamp(0.0, 1.0).cpu()]

    if step_count > 1:
        refine_sigmas = torch.linspace(sigma_max, sigma_min, step_count + 1)[1:-1]
        for sigma in refine_sigmas:
            sigma_val = float(sigma.item())
            std = (sigma_val**2 - sigma_min**2) ** 0.5
            xt = x + std * torch.randn_like(x)
            x = model(xt, torch.full((int(num_samples),), sigma_val, device=device))
            frames.append(x.view(-1, 1, 28, 28).clamp(0.0, 1.0).cpu())

    if return_all:
        return torch.stack(frames, dim=0)
    return frames[-1]
