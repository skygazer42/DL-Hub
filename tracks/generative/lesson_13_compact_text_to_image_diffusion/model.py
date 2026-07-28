from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    image_size: int = 28
    in_channels: int = 1
    hidden_channels: int = 32
    time_embed_dim: int = 32
    text_vocab_size: int = 4
    text_embed_dim: int = 16


@dataclass(frozen=True)
class DiffusionSchedule:
    num_steps: int = 20
    beta_start: float = 1e-4
    beta_end: float = 0.02

    def betas(self) -> torch.Tensor:
        return torch.linspace(float(self.beta_start), float(self.beta_end), int(self.num_steps), dtype=torch.float32)

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
        emb = torch.cat(
            [emb, torch.zeros((emb.shape[0], dim - emb.shape[1]), device=timesteps.device, dtype=emb.dtype)],
            dim=1,
        )
    return emb[:, :dim]


def q_sample(schedule: DiffusionSchedule, x0: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    alpha_bar = schedule.alpha_bars().to(device=x0.device, dtype=x0.dtype)
    gathered = alpha_bar.index_select(0, timesteps)
    while gathered.ndim < x0.ndim:
        gathered = gathered.unsqueeze(-1)
    return gathered.sqrt() * x0 + (1.0 - gathered).sqrt() * noise


class CompactTextConditionedDenoiser(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        hidden = int(cfg.hidden_channels)
        self.text_embed = nn.Embedding(int(cfg.text_vocab_size), int(cfg.text_embed_dim))
        self.time_proj = nn.Sequential(
            nn.Linear(int(cfg.time_embed_dim), int(cfg.time_embed_dim)),
            nn.SiLU(),
            nn.Linear(int(cfg.time_embed_dim), int(cfg.time_embed_dim)),
        )
        self.net = nn.Sequential(
            nn.Conv2d(int(cfg.in_channels) + 2, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden, int(cfg.in_channels), kernel_size=3, padding=1),
        )

    def forward(self, xt: torch.Tensor, token_ids: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        b, _, h, w = xt.shape
        t_emb = timestep_embedding(timesteps, int(self.cfg.time_embed_dim))
        t_emb = self.time_proj(t_emb).to(dtype=xt.dtype)
        t_map = t_emb.mean(dim=1, keepdim=True).view(b, 1, 1, 1).expand(b, 1, h, w)

        txt_emb = self.text_embed(token_ids).to(dtype=xt.dtype)
        txt_map = txt_emb.mean(dim=1, keepdim=True).view(b, 1, 1, 1).expand(b, 1, h, w)

        x = torch.cat([xt, t_map, txt_map], dim=1)
        return self.net(x)

    @torch.no_grad()
    def sample(
        self,
        schedule: DiffusionSchedule,
        *,
        token_ids: torch.Tensor,
        device: torch.device | str,
        num_steps: int | None = None,
        return_all: bool = False,
    ) -> torch.Tensor:
        device = torch.device(device)
        token_ids = token_ids.to(device=device, dtype=torch.long)
        n = int(token_ids.shape[0])
        shape = (n, int(self.cfg.in_channels), int(self.cfg.image_size), int(self.cfg.image_size))

        step_count = int(schedule.num_steps if num_steps is None else min(num_steps, schedule.num_steps))
        betas = schedule.betas().to(device=device)
        alphas = schedule.alphas().to(device=device)
        alpha_bars = schedule.alpha_bars().to(device=device)

        xt = torch.randn(shape, device=device)
        frames = [xt.detach().cpu()] if return_all else None

        for step in range(step_count - 1, -1, -1):
            t = torch.full((n,), step, device=device, dtype=torch.long)
            pred_noise = self(xt, token_ids, t)
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
