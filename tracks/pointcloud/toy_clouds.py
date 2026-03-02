from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from dlhub.data.splits import train_val_split_indices


def _sample_cube(*, num_points: int, g: torch.Generator) -> torch.Tensor:
    # Uniform points inside a cube.
    pts = torch.rand((num_points, 3), generator=g, dtype=torch.float32) * 2.0 - 1.0
    return pts


def _sample_sphere(*, num_points: int, g: torch.Generator, noise_std: float = 0.02) -> torch.Tensor:
    # Sample points near the unit sphere surface.
    pts = torch.randn((num_points, 3), generator=g, dtype=torch.float32)
    pts = pts / pts.norm(dim=1, keepdim=True).clamp(min=1e-8)
    noise = torch.randn(pts.shape, generator=g, dtype=torch.float32)
    pts = pts + noise * float(noise_std)
    return pts


@dataclass(frozen=True)
class DataConfig:
    num_samples: int = 2048
    num_points: int = 128
    batch_size: int = 64
    val_fraction: float = 0.2
    seed: int = 0
    num_workers: int = 0


class ToyPointCloudDataset(Dataset):
    """Cube vs Sphere point cloud classification dataset (fully synthetic)."""

    def __init__(self, cfg: DataConfig) -> None:
        self.cfg = cfg
        num_samples = int(cfg.num_samples)
        num_points = int(cfg.num_points)

        g = torch.Generator().manual_seed(int(cfg.seed))
        labels = torch.randint(low=0, high=2, size=(num_samples,), generator=g, dtype=torch.long)

        clouds = torch.empty((num_samples, num_points, 3), dtype=torch.float32)
        for i in range(num_samples):
            if int(labels[i].item()) == 0:
                clouds[i] = _sample_cube(num_points=num_points, g=g)
            else:
                clouds[i] = _sample_sphere(num_points=num_points, g=g)

        self.clouds = clouds
        self.labels = labels

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        i = int(idx)
        return self.clouds[i], self.labels[i]


def get_dataloaders(cfg: DataConfig) -> tuple[DataLoader, DataLoader]:
    ds = ToyPointCloudDataset(cfg)
    train_idx, val_idx = train_val_split_indices(n=len(ds), val_fraction=cfg.val_fraction, seed=cfg.seed)

    train_ds = Subset(ds, train_idx)
    val_ds = Subset(ds, val_idx)

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg.batch_size),
        shuffle=True,
        num_workers=int(cfg.num_workers),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg.batch_size),
        shuffle=False,
        num_workers=int(cfg.num_workers),
        drop_last=False,
    )
    return train_loader, val_loader


__all__ = ["DataConfig", "ToyPointCloudDataset", "get_dataloaders"]

