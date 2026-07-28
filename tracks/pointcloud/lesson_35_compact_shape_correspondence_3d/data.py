from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from dlhub.data.splits import train_val_split_indices


def _sample_cube_surface(*, num_points: int, g: torch.Generator) -> torch.Tensor:
    face = torch.randint(0, 6, (num_points,), generator=g, dtype=torch.long)
    points = torch.rand((num_points, 3), generator=g, dtype=torch.float32) * 2.0 - 1.0
    axis = face // 2
    sign = (face % 2) * 2 - 1
    for axis_id in (0, 1, 2):
        mask = axis == axis_id
        if int(mask.sum().item()) > 0:
            points[mask, axis_id] = sign[mask].to(torch.float32)
    return points


def _sample_sphere_surface(*, num_points: int, g: torch.Generator) -> torch.Tensor:
    points = torch.randn((num_points, 3), generator=g, dtype=torch.float32)
    return points / points.norm(dim=1, keepdim=True).clamp(min=1e-8)


def _rotation_matrix(theta_y: float, theta_z: float) -> torch.Tensor:
    cy, sy = math.cos(theta_y), math.sin(theta_y)
    cz, sz = math.cos(theta_z), math.sin(theta_z)
    ry = torch.tensor([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=torch.float32)
    rz = torch.tensor([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32)
    return rz @ ry


@dataclass(frozen=True)
class DataConfig:
    num_samples: int = 256
    num_points: int = 96
    batch_size: int = 16
    val_fraction: float = 0.2
    seed: int = 0
    num_workers: int = 0
    translation_scale: float = 0.2
    noise_std: float = 0.01


class SyntheticShapeCorrespondenceDataset(Dataset):
    def __init__(self, cfg: DataConfig) -> None:
        self.cfg = cfg
        if int(cfg.num_points) < 16:
            raise ValueError("num_points must be >= 16")
        if not (0.0 < float(cfg.val_fraction) < 1.0):
            raise ValueError("val_fraction must be in (0, 1)")
        if float(cfg.translation_scale) < 0.0:
            raise ValueError("translation_scale must be >= 0")
        if float(cfg.noise_std) < 0.0:
            raise ValueError("noise_std must be >= 0")

    def __len__(self) -> int:
        return int(self.cfg.num_samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sample_idx = int(idx)
        cfg = self.cfg
        num_points = int(cfg.num_points)
        g = torch.Generator().manual_seed(int(cfg.seed) * 1_000_003 + sample_idx)

        if bool(torch.rand((), generator=g).item() < 0.5):
            source = _sample_cube_surface(num_points=num_points, g=g)
        else:
            source = _sample_sphere_surface(num_points=num_points, g=g)

        phase = 2.0 * math.pi * ((sample_idx % 41) / 41.0)
        rot = _rotation_matrix(theta_y=0.45 * math.sin(phase), theta_z=0.55 * math.cos(phase))
        transformed = source @ rot.T
        translation = torch.tensor(
            [math.cos(phase), 0.5 * math.sin(phase), math.cos(0.5 * phase)],
            dtype=torch.float32,
        ) * float(cfg.translation_scale)
        transformed = transformed + translation.unsqueeze(0)

        perm = torch.randperm(num_points, generator=g)
        target = transformed[perm]
        if float(cfg.noise_std) > 0.0:
            target = target + torch.randn((num_points, 3), generator=g, dtype=torch.float32) * float(
                cfg.noise_std
            )

        correspondence = torch.empty((num_points,), dtype=torch.long)
        correspondence[perm] = torch.arange(num_points, dtype=torch.long)
        return source, target, correspondence


def get_dataloaders(cfg: DataConfig) -> tuple[DataLoader, DataLoader]:
    dataset = SyntheticShapeCorrespondenceDataset(cfg)
    train_idx, val_idx = train_val_split_indices(
        n=len(dataset), val_fraction=float(cfg.val_fraction), seed=int(cfg.seed)
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


__all__ = ["DataConfig", "SyntheticShapeCorrespondenceDataset", "get_dataloaders"]
