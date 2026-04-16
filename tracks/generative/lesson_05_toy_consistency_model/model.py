from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    hidden_dim: int = 128
    time_embed_dim: int = 32


@dataclass(frozen=True)
class DiffusionSchedule:
    num_steps: int = 20
    beta_start: float = 1e-4
    beta_end: float = 0.02

    def betas(self) -> torch.Tensor:
        return torch.linspace(
            float(self.beta_start), float(self.beta_end), int(self.num_steps), dtype=torch.float32
        )

    def alphas(self) -> torch.Tensor:
        return 1.0 - self.betas()

    def alpha_bars(self) -> torch.Tensor:
        return torch.cumprod(self.alphas(), dim=0)


def timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    half_dim = max(1, dim // 2)
    device = timesteps.device
    exponent = torch.arange(half_dim, device=device, dtype=torch.float32) / float(half_dim)
    frequencies = torch.exp(-torch.log(torch.tensor(10000.0, device=device)) * exponent)
    angles = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
    emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
    if emb.shape[1] < dim:
        emb = torch.cat([emb, torch.zeros((emb.shape[0], dim - emb.shape[1]), device=device)], dim=1)
    return emb[:, :dim]


def q_sample(
    schedule: DiffusionSchedule,
    x0: torch.Tensor,
    timesteps: torch.Tensor,
    noise: torch.Tensor,
) -> torch.Tensor:
    alpha_bar = schedule.alpha_bars().to(device=x0.device, dtype=x0.dtype)
    gathered = alpha_bar.index_select(0, timesteps)
    while gathered.ndim < x0.ndim:
        gathered = gathered.unsqueeze(-1)
    return gathered.sqrt() * x0 + (1.0 - gathered).sqrt() * noise


class DiffusionMLP(nn.Module):
    """A tiny denoiser over flattened 28x28 images."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        in_dim = 28 * 28
        self.net = nn.Sequential(
            nn.Linear(in_dim + cfg.time_embed_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, in_dim),
        )

    def forward(self, xt: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        time_emb = timestep_embedding(timesteps, self.cfg.time_embed_dim).to(dtype=xt.dtype)
        return self.net(torch.cat([xt, time_emb], dim=1))


@torch.no_grad()
def sample_reverse_diffusion(
    model: DiffusionMLP,
    schedule: DiffusionSchedule,
    *,
    num_samples: int,
    device: torch.device,
    num_steps: int | None = None,
    return_all: bool = False,
) -> torch.Tensor:
    step_count = int(schedule.num_steps if num_steps is None else min(num_steps, schedule.num_steps))
    betas = schedule.betas().to(device=device)
    alphas = schedule.alphas().to(device=device)
    alpha_bars = schedule.alpha_bars().to(device=device)

    xt = torch.randn((int(num_samples), 28 * 28), device=device)
    frames = [xt.view(-1, 1, 28, 28).cpu()]

    for step in range(step_count - 1, -1, -1):
        t = torch.full((int(num_samples),), step, device=device, dtype=torch.long)
        pred_noise = model(xt, t)

        alpha_t = alphas[step]
        alpha_bar_t = alpha_bars[step]
        beta_t = betas[step]
        coef = beta_t / torch.sqrt(torch.clamp(1.0 - alpha_bar_t, min=1e-6))
        mean = (xt - coef * pred_noise) / torch.sqrt(alpha_t)

        if step > 0:
            xt = mean + torch.sqrt(beta_t) * torch.randn_like(xt)
        else:
            xt = mean

        if return_all:
            frames.append(xt.view(-1, 1, 28, 28).cpu())

    images = xt.view(-1, 1, 28, 28).clamp(0.0, 1.0)
    if return_all:
        return torch.stack(frames[1:] + [images.cpu()], dim=0)
    return images.cpu()
