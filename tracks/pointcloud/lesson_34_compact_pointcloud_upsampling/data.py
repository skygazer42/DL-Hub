from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from dlhub.data.splits import train_val_split_indices
from tracks.pointcloud.synthetic_clouds import _sample_cube_surface, _sample_sphere


@dataclass(frozen=True)
class DataConfig:
    num_samples: int = 2048
    num_sparse_points: int = 64
    upsample_factor: int = 2
    batch_size: int = 64
    val_fraction: float = 0.2
    seed: int = 0
    num_workers: int = 0
    p_sphere: float = 0.5
    sphere_surface_noise: float = 0.0
    cube_surface_noise: float = 0.0
    shuffle_points: bool = True


class SyntheticPointCloudUpsamplingDataset(Dataset):
    """Synthetic sparse-to-dense pointcloud upsampling pairs."""

    def __init__(self, cfg: DataConfig) -> None:
        self.cfg = cfg
        if int(cfg.num_sparse_points) < 16:
            raise ValueError("num_sparse_points must be >= 16")
        if int(cfg.upsample_factor) < 2:
            raise ValueError("upsample_factor must be >= 2")
        if not (0.0 < float(cfg.val_fraction) < 1.0):
            raise ValueError("val_fraction must be in (0, 1)")
        if not (0.0 <= float(cfg.p_sphere) <= 1.0):
            raise ValueError("p_sphere must be in [0, 1]")

    def __len__(self) -> int:
        return int(self.cfg.num_samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = self.cfg
        sample_seed = int(cfg.seed) * 1_000_003 + int(idx)
        g = torch.Generator().manual_seed(sample_seed)

        n_sparse = int(cfg.num_sparse_points)
        n_dense = n_sparse * int(cfg.upsample_factor)
        is_sphere = bool(torch.rand((), generator=g).item() < float(cfg.p_sphere))

        if is_sphere:
            dense = _sample_sphere(
                num_points=n_dense,
                g=g,
                noise_std=float(cfg.sphere_surface_noise),
            )
        else:
            dense = _sample_cube_surface(
                num_points=n_dense,
                g=g,
                noise_std=float(cfg.cube_surface_noise),
            )

        sparse_idx = torch.randperm(n_dense, generator=g)[:n_sparse]
        sparse = dense[sparse_idx]

        if bool(cfg.shuffle_points):
            dense = dense[torch.randperm(n_dense, generator=g)]
            sparse = sparse[torch.randperm(n_sparse, generator=g)]

        return sparse.to(torch.float32), dense.to(torch.float32)


def get_dataloaders(cfg: DataConfig) -> tuple[DataLoader, DataLoader]:
    dataset = SyntheticPointCloudUpsamplingDataset(cfg)
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


__all__ = ["DataConfig", "SyntheticPointCloudUpsamplingDataset", "get_dataloaders"]
