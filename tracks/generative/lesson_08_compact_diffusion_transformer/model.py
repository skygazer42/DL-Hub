from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    image_size: int = 28
    patch_size: int = 4
    in_channels: int = 1
    hidden_dim: int = 64
    depth: int = 2
    num_heads: int = 4
    mlp_ratio: float = 2.0


@dataclass(frozen=True)
class DiffusionSchedule:
    num_steps: int = 20
    beta_start: float = 1e-4
    beta_end: float = 0.02

    def betas(self) -> torch.Tensor:
        return torch.linspace(
            float(self.beta_start),
            float(self.beta_end),
            int(self.num_steps),
            dtype=torch.float32,
        )

    def alphas(self) -> torch.Tensor:
        return 1.0 - self.betas()

    def alpha_bars(self) -> torch.Tensor:
        return torch.cumprod(self.alphas(), dim=0)


def timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    half_dim = max(1, dim // 2)
    exponent = torch.arange(half_dim, device=timesteps.device, dtype=torch.float32) / float(half_dim)
    frequencies = torch.exp(-torch.log(torch.tensor(10000.0, device=timesteps.device)) * exponent)
    angles = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
    emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
    if emb.shape[1] < dim:
        pad = torch.zeros((emb.shape[0], dim - emb.shape[1]), device=timesteps.device)
        emb = torch.cat([emb, pad], dim=1)
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


class DiTTiny(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        if cfg.image_size % cfg.patch_size != 0:
            raise ValueError(
                f"image_size ({cfg.image_size}) must be divisible by patch_size ({cfg.patch_size})"
            )
        if cfg.hidden_dim % cfg.num_heads != 0:
            raise ValueError(
                f"hidden_dim ({cfg.hidden_dim}) must be divisible by num_heads ({cfg.num_heads})"
            )

        self.grid_size = cfg.image_size // cfg.patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.patch_dim = cfg.in_channels * cfg.patch_size * cfg.patch_size

        self.patch_embed = nn.Conv2d(
            cfg.in_channels,
            cfg.hidden_dim,
            kernel_size=cfg.patch_size,
            stride=cfg.patch_size,
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, cfg.hidden_dim))
        self.time_proj = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden_dim,
            nhead=cfg.num_heads,
            dim_feedforward=int(cfg.hidden_dim * cfg.mlp_ratio),
            activation="gelu",
            batch_first=True,
            dropout=0.0,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.depth)
        self.out_proj = nn.Linear(cfg.hidden_dim, self.patch_dim)

    def _unpatchify(self, patches: torch.Tensor) -> torch.Tensor:
        batch_size = patches.shape[0]
        p = self.cfg.patch_size
        c = self.cfg.in_channels
        g = self.grid_size
        x = patches.view(batch_size, g, g, c, p, p)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        return x.view(batch_size, c, g * p, g * p)

    def forward(self, xt: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        tokens = self.patch_embed(xt).flatten(2).transpose(1, 2)
        t_embed = self.time_proj(timestep_embedding(timesteps, self.cfg.hidden_dim).to(dtype=xt.dtype))
        tokens = tokens + self.pos_embed.to(dtype=xt.dtype) + t_embed.unsqueeze(1)
        hidden = self.encoder(tokens)
        pred_patches = self.out_proj(hidden)
        return self._unpatchify(pred_patches)

    @torch.no_grad()
    def sample(
        self,
        schedule: DiffusionSchedule,
        *,
        num_samples: int,
        device: torch.device | str,
        num_steps: int | None = None,
        return_all: bool = False,
    ) -> torch.Tensor:
        device = torch.device(device)
        step_count = int(schedule.num_steps if num_steps is None else min(num_steps, schedule.num_steps))
        betas = schedule.betas().to(device=device)
        alphas = schedule.alphas().to(device=device)
        alpha_bars = schedule.alpha_bars().to(device=device)

        xt = torch.randn(
            (int(num_samples), self.cfg.in_channels, self.cfg.image_size, self.cfg.image_size), device=device
        )
        frames = [xt.detach().cpu()] if return_all else None

        for step in range(step_count - 1, -1, -1):
            t = torch.full((int(num_samples),), step, device=device, dtype=torch.long)
            pred_noise = self(xt, t)

            alpha_t = alphas[step]
            alpha_bar_t = alpha_bars[step]
            beta_t = betas[step]
            coef = beta_t / torch.sqrt(torch.clamp(1.0 - alpha_bar_t, min=1e-6))
            mean = (xt - coef * pred_noise) / torch.sqrt(alpha_t)

            if step > 0:
                xt = mean + torch.sqrt(beta_t) * torch.randn_like(xt)
            else:
                xt = mean

            if frames is not None:
                frames.append(xt.detach().cpu())

        samples = xt.clamp(0.0, 1.0).cpu()
        if frames is None:
            return samples
        return torch.stack(frames[1:] + [samples], dim=0)
