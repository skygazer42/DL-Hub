from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from dlhub.data.splits import train_val_split_indices
from tracks.pointcloud.toy_clouds import _sample_cube_surface, _sample_sphere


@dataclass(frozen=True)
class DataConfig:
    num_samples: int = 256
    sequence_length: int = 4
    forecast_horizon: int = 2
    num_points: int = 96
    batch_size: int = 16
    val_fraction: float = 0.2
    seed: int = 0
    num_workers: int = 0
    motion_scale: float = 0.18
    jitter_std: float = 0.01
    p_sphere: float = 0.5


def _sample_base_cloud(cfg: DataConfig, g: torch.Generator) -> torch.Tensor:
    num_points = int(cfg.num_points)
    if bool(torch.rand((), generator=g).item() < float(cfg.p_sphere)):
        return _sample_sphere(num_points=num_points, g=g, noise_std=0.01)
    return _sample_cube_surface(num_points=num_points, g=g, noise_std=0.0)


class SyntheticPointCloudForecastingDataset(Dataset):
    """Synthetic moving pointcloud sequences with future supervision."""

    def __init__(self, cfg: DataConfig) -> None:
        self.cfg = cfg
        if int(cfg.num_samples) <= 0:
            raise ValueError("num_samples must be > 0")
        if int(cfg.sequence_length) < 2:
            raise ValueError("sequence_length must be >= 2")
        if int(cfg.forecast_horizon) < 1:
            raise ValueError("forecast_horizon must be >= 1")
        if int(cfg.num_points) < 16:
            raise ValueError("num_points must be >= 16")
        if not (0.0 < float(cfg.val_fraction) < 1.0):
            raise ValueError("val_fraction must be in (0, 1)")
        if float(cfg.motion_scale) < 0.0:
            raise ValueError("motion_scale must be >= 0")
        if float(cfg.jitter_std) < 0.0:
            raise ValueError("jitter_std must be >= 0")
        if not (0.0 <= float(cfg.p_sphere) <= 1.0):
            raise ValueError("p_sphere must be in [0, 1]")

    def __len__(self) -> int:
        return int(self.cfg.num_samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        cfg = self.cfg
        sample_idx = int(idx)
        total_steps = int(cfg.sequence_length) + int(cfg.forecast_horizon)
        g = torch.Generator().manual_seed(int(cfg.seed) * 1_000_003 + sample_idx)

        base = _sample_base_cloud(cfg, g)
        phase = 2.0 * math.pi * ((sample_idx % 37) / 37.0)
        velocity = torch.tensor(
            [math.cos(phase), math.sin(phase), 0.5 * math.cos(0.5 * phase)],
            dtype=torch.float32,
        ) * float(cfg.motion_scale)
        deformation = torch.stack((base[:, 1], -base[:, 0], 0.4 * base[:, 2]), dim=-1)

        sequence: list[torch.Tensor] = []
        for step in range(total_steps):
            step_ratio = float(step) / max(1.0, float(total_steps - 1))
            cloud = base + float(step) * velocity.unsqueeze(0)
            cloud = cloud + (0.12 * float(cfg.motion_scale) * step_ratio) * deformation
            if float(cfg.jitter_std) > 0.0:
                noise = torch.randn(base.shape, generator=g, dtype=torch.float32) * float(cfg.jitter_std)
                cloud = cloud + noise
            sequence.append(cloud.to(torch.float32))

        stacked = torch.stack(sequence, dim=0)
        history = stacked[: int(cfg.sequence_length)]
        future = stacked[int(cfg.sequence_length) :]
        return history, {"future": future}


def get_dataloaders(cfg: DataConfig) -> tuple[DataLoader, DataLoader]:
    ds = SyntheticPointCloudForecastingDataset(cfg)
    train_idx, val_idx = train_val_split_indices(
        n=len(ds),
        val_fraction=float(cfg.val_fraction),
        seed=int(cfg.seed),
    )
    train_loader = DataLoader(
        Subset(ds, train_idx),
        batch_size=int(cfg.batch_size),
        shuffle=True,
        num_workers=int(cfg.num_workers),
        drop_last=False,
    )
    val_loader = DataLoader(
        Subset(ds, val_idx),
        batch_size=int(cfg.batch_size),
        shuffle=False,
        num_workers=int(cfg.num_workers),
        drop_last=False,
    )
    return train_loader, val_loader


__all__ = ["DataConfig", "SyntheticPointCloudForecastingDataset", "get_dataloaders"]
