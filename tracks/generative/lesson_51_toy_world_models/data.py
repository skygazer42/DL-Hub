from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from dlhub.data.splits import train_val_split_indices


@dataclass(frozen=True)
class DataConfig:
    num_samples: int = 128
    batch_size: int = 8
    image_size: int = 16
    in_channels: int = 3
    action_dim: int = 4
    context_dim: int = 12
    seed: int = 0
    num_workers: int = 0
    val_fraction: float = 0.2


def _render_blob(image_size: int, center_x: float, center_y: float, intensity: float) -> torch.Tensor:
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, image_size, dtype=torch.float32),
        torch.linspace(-1.0, 1.0, image_size, dtype=torch.float32),
        indexing="ij",
    )
    sigma = 0.18
    dist = (xx - center_x).pow(2) + (yy - center_y).pow(2)
    blob = float(intensity) * torch.exp(-dist / (2.0 * sigma * sigma))
    background = 0.08 + 0.03 * torch.sin(2.0 * xx - 1.5 * yy)
    return (blob + background).clamp(0.0, 1.0)


def _make_transition(cfg: DataConfig, sample_idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    generator = torch.Generator().manual_seed(int(cfg.seed) * 1_000_003 + int(sample_idx))
    image_size = int(cfg.image_size)
    in_channels = int(cfg.in_channels)
    action_dim = int(cfg.action_dim)
    context_dim = int(cfg.context_dim)

    phase = (2.0 * math.pi * (sample_idx % 67)) / 67.0
    cx = float(0.55 * math.sin(phase))
    cy = float(0.55 * math.cos(phase))
    action = torch.randn((action_dim,), generator=generator, dtype=torch.float32) * 0.35
    prompt = torch.randn((context_dim,), generator=generator, dtype=torch.float32) * 0.25

    dx = float(0.22 * torch.tanh(action[0]).item())
    dy = float(0.22 * torch.tanh(action[1]).item())
    brightness_shift = float(0.18 * torch.tanh(prompt.mean()).item())

    obs_channels: list[torch.Tensor] = []
    next_obs_channels: list[torch.Tensor] = []
    for channel_id in range(in_channels):
        intensity = 0.55 + 0.08 * channel_id
        obs_frame = _render_blob(image_size, cx, cy, intensity)
        next_frame = _render_blob(
            image_size,
            max(-0.9, min(0.9, cx + dx)),
            max(-0.9, min(0.9, cy + dy)),
            max(0.15, min(0.95, intensity + brightness_shift)),
        )
        obs_frame = obs_frame + 0.01 * torch.rand((image_size, image_size), generator=generator, dtype=torch.float32)
        next_frame = next_frame + 0.01 * torch.rand(
            (image_size, image_size), generator=generator, dtype=torch.float32
        )
        obs_channels.append(obs_frame.clamp(0.0, 1.0))
        next_obs_channels.append(next_frame.clamp(0.0, 1.0))

    obs = torch.stack(obs_channels, dim=0).to(torch.float32)
    next_obs = torch.stack(next_obs_channels, dim=0).to(torch.float32)

    speed = float((dx**2 + dy**2) ** 0.5)
    reward_value = max(0.0, 1.0 - speed - 0.35 * abs(brightness_shift))
    done_value = 1.0 if (abs(cx + dx) > 0.84 or abs(cy + dy) > 0.84) else 0.0

    targets = {
        "next_obs": next_obs,
        "reward": torch.tensor([reward_value], dtype=torch.float32),
        "done": torch.tensor([done_value], dtype=torch.float32),
    }
    return obs, action.to(torch.float32), prompt.to(torch.float32), targets


class ToyWorldModelsDataset(Dataset):
    def __init__(self, cfg: DataConfig) -> None:
        self.cfg = cfg
        if int(cfg.num_samples) < 4:
            raise ValueError("num_samples must be >= 4")
        if int(cfg.image_size) < 8:
            raise ValueError("image_size must be >= 8")
        if int(cfg.in_channels) <= 0:
            raise ValueError("in_channels must be positive")
        if int(cfg.action_dim) <= 0:
            raise ValueError("action_dim must be positive")
        if int(cfg.context_dim) <= 0:
            raise ValueError("context_dim must be positive")
        if not (0.0 < float(cfg.val_fraction) < 1.0):
            raise ValueError("val_fraction must be in (0, 1)")

    def __len__(self) -> int:
        return int(self.cfg.num_samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        return _make_transition(self.cfg, int(idx))


def get_dataloaders(cfg: DataConfig) -> tuple[DataLoader, DataLoader]:
    dataset: Dataset = ToyWorldModelsDataset(cfg)
    train_idx, val_idx = train_val_split_indices(
        n=len(dataset),
        val_fraction=float(cfg.val_fraction),
        seed=int(cfg.seed),
    )
    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=int(cfg.batch_size),
        shuffle=True,
        num_workers=int(cfg.num_workers),
        drop_last=False,
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=int(cfg.batch_size),
        shuffle=False,
        num_workers=int(cfg.num_workers),
        drop_last=False,
    )
    return train_loader, val_loader


__all__ = ["DataConfig", "ToyWorldModelsDataset", "get_dataloaders"]

