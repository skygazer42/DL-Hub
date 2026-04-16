import math
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    image_size: int = 28
    in_channels: int = 1
    hidden_channels: int = 24
    time_embed_dim: int = 16


def sample_time(batch_size: int, *, device: torch.device | str | None = None) -> torch.Tensor:
    return torch.rand((batch_size,), device=device, dtype=torch.float32)


def build_rectified_targets(
    images: torch.Tensor,
    noise: torch.Tensor,
    times: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    t = times.view(-1, 1, 1, 1)
    xt = (1.0 - t) * noise + t * images
    # Rectified flow still learns a straight-line velocity for paired endpoints.
    target_velocity = images - noise
    return xt, target_velocity


def _time_features(times: torch.Tensor, embed_dim: int) -> torch.Tensor:
    if embed_dim % 2 != 0:
        raise ValueError(f"time_embed_dim must be even, got {embed_dim}")
    half = embed_dim // 2
    freqs = torch.exp(
        torch.linspace(0.0, math.log(1000.0), half, device=times.device, dtype=times.dtype)
    )
    angles = times.unsqueeze(1) * freqs.unsqueeze(0)
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)


class RectifiedFlowModel(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        if cfg.image_size != 28:
            raise ValueError("This toy lesson currently supports 28x28 images only.")

        self.time_mlp = nn.Sequential(
            nn.Linear(cfg.time_embed_dim, cfg.hidden_channels),
            nn.SiLU(),
            nn.Linear(cfg.hidden_channels, cfg.hidden_channels),
        )
        self.in_proj = nn.Conv2d(cfg.in_channels, cfg.hidden_channels, kernel_size=3, padding=1)
        self.net = nn.Sequential(
            nn.SiLU(),
            nn.Conv2d(cfg.hidden_channels, cfg.hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(cfg.hidden_channels, cfg.hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(cfg.hidden_channels, cfg.in_channels, kernel_size=3, padding=1),
        )

    def forward(self, xt: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        if times.ndim != 1:
            raise ValueError(f"Expected `times` shape (B,), got {tuple(times.shape)}")
        time_embed = self.time_mlp(_time_features(times, self.cfg.time_embed_dim)).unsqueeze(-1).unsqueeze(-1)
        hidden = self.in_proj(xt) + time_embed
        return self.net(hidden)

    @torch.no_grad()
    def sample(
        self,
        *,
        num_samples: int,
        device: torch.device | str,
        num_steps: int = 16,
        return_trajectory: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        device = torch.device(device)
        state = torch.randn(
            (num_samples, self.cfg.in_channels, self.cfg.image_size, self.cfg.image_size),
            device=device,
        )
        time_grid = torch.linspace(0.0, 1.0, num_steps + 1, device=device)
        trajectory = [state.detach().cpu()] if return_trajectory else None

        for step in range(num_steps):
            times = torch.full((num_samples,), float(time_grid[step].item()), device=device)
            velocity = self(state, times)
            dt = float((time_grid[step + 1] - time_grid[step]).item())
            state = state + dt * velocity
            if trajectory is not None:
                trajectory.append(state.detach().cpu())

        samples = state.clamp(0.0, 1.0)
        if trajectory is None:
            return samples
        return samples, torch.stack(trajectory, dim=0)
