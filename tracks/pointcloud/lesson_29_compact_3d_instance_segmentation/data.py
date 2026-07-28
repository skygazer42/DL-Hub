import math
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from dlhub.data.splits import train_val_split_indices


@dataclass(frozen=True)
class DataConfig:
    num_samples: int = 256
    num_points: int = 96
    batch_size: int = 16
    val_fraction: float = 0.2
    seed: int = 0
    num_workers: int = 0
    cluster_offset: float = 0.9
    cluster_std: float = 0.08


class SyntheticInstanceSegmentation3DDataset(Dataset):
    def __init__(self, cfg: DataConfig) -> None:
        self.cfg = cfg
        if int(cfg.num_points) < 16:
            raise ValueError("num_points must be >= 16")
        if not (0.0 < float(cfg.val_fraction) < 1.0):
            raise ValueError("val_fraction must be in (0, 1)")
        if float(cfg.cluster_offset) <= 0.0:
            raise ValueError("cluster_offset must be > 0")
        if float(cfg.cluster_std) <= 0.0:
            raise ValueError("cluster_std must be > 0")

    def __len__(self) -> int:
        return int(self.cfg.num_samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample_idx = int(idx)
        cfg = self.cfg
        num_points = int(cfg.num_points)
        g = torch.Generator().manual_seed(int(cfg.seed) * 1_000_003 + sample_idx)

        split = num_points // 2
        count0 = split
        count1 = num_points - split

        phase = 2.0 * math.pi * ((sample_idx % 41) / 41.0)
        drift = 0.2 * float(cfg.cluster_offset) * torch.tensor(
            [math.cos(phase), math.sin(phase), 0.25 * math.cos(0.5 * phase)],
            dtype=torch.float32,
        )
        offset = float(cfg.cluster_offset)
        center0 = torch.tensor([-offset, 0.0, 0.0], dtype=torch.float32) + drift
        center1 = torch.tensor([offset, 0.0, 0.0], dtype=torch.float32) + drift

        std = float(cfg.cluster_std)
        points0 = center0 + std * torch.randn((count0, 3), generator=g, dtype=torch.float32)
        points1 = center1 + std * torch.randn((count1, 3), generator=g, dtype=torch.float32)

        points = torch.cat([points0, points1], dim=0)
        instance_ids = torch.cat(
            [
                torch.zeros(count0, dtype=torch.long),
                torch.ones(count1, dtype=torch.long),
            ],
            dim=0,
        )
        perm = torch.randperm(num_points, generator=g)
        return points[perm], instance_ids[perm]


def get_dataloaders(cfg: DataConfig) -> tuple[DataLoader, DataLoader]:
    dataset = SyntheticInstanceSegmentation3DDataset(cfg)
    train_idx, val_idx = train_val_split_indices(
        n=len(dataset), val_fraction=cfg.val_fraction, seed=cfg.seed
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


__all__ = ["DataConfig", "SyntheticInstanceSegmentation3DDataset", "get_dataloaders"]
