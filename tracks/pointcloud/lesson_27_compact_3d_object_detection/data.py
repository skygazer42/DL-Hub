import math
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from dlhub.data.splits import train_val_split_indices


def _rotate_z(points: torch.Tensor, yaw: float) -> torch.Tensor:
    c = math.cos(float(yaw))
    s = math.sin(float(yaw))
    rot = torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32)
    return points @ rot.T


def _sample_unit_sphere(*, n: int, g: torch.Generator) -> torch.Tensor:
    points = torch.randn((n, 3), generator=g, dtype=torch.float32)
    return points / points.norm(dim=1, keepdim=True).clamp(min=1e-8)


@dataclass(frozen=True)
class DataConfig:
    num_samples: int = 512
    num_points: int = 128
    batch_size: int = 16
    val_fraction: float = 0.2
    seed: int = 0
    num_workers: int = 0
    noise_points: int = 32


class SyntheticObjectDetection3DDataset(Dataset):
    def __init__(self, cfg: DataConfig) -> None:
        self.cfg = cfg
        if int(cfg.num_points) < 32:
            raise ValueError("num_points must be >= 32")
        if int(cfg.noise_points) < 0:
            raise ValueError("noise_points must be >= 0")
        if int(cfg.noise_points) >= int(cfg.num_points):
            raise ValueError("noise_points must be < num_points")
        if not (0.0 < float(cfg.val_fraction) < 1.0):
            raise ValueError("val_fraction must be in (0, 1)")

    def __len__(self) -> int:
        return int(self.cfg.num_samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cfg = self.cfg
        sample_idx = int(idx)
        g = torch.Generator().manual_seed(int(cfg.seed) * 1_000_003 + sample_idx)

        num_points = int(cfg.num_points)
        noise_points = int(cfg.noise_points)
        object_points = num_points - noise_points

        label = int(torch.randint(0, 2, (1,), generator=g, dtype=torch.long).item())
        center = torch.rand((3,), generator=g, dtype=torch.float32) * 1.2 - 0.6
        dims = torch.rand((3,), generator=g, dtype=torch.float32) * 0.45 + 0.3
        yaw = float(torch.rand((), generator=g, dtype=torch.float32).item() * (2.0 * math.pi) - math.pi)

        if label == 0:
            local = (torch.rand((object_points, 3), generator=g, dtype=torch.float32) * 2.0 - 1.0) * (
                0.5 * dims
            ).unsqueeze(0)
        else:
            local = _sample_unit_sphere(n=object_points, g=g) * (0.5 * dims).unsqueeze(0)

        object_cloud = _rotate_z(local, yaw) + center.unsqueeze(0)
        clutter = torch.rand((noise_points, 3), generator=g, dtype=torch.float32) * 3.2 - 1.6
        points = torch.cat([object_cloud, clutter], dim=0)
        perm = torch.randperm(num_points, generator=g)
        points = points[perm]

        points = points + 0.01 * torch.randn(points.shape, generator=g, dtype=torch.float32)
        box = torch.tensor(
            [center[0], center[1], center[2], dims[0], dims[1], dims[2], yaw], dtype=torch.float32
        )
        label_tensor = torch.tensor(label, dtype=torch.long)
        return points.to(torch.float32), box, label_tensor


def get_dataloaders(cfg: DataConfig) -> tuple[DataLoader, DataLoader]:
    dataset = SyntheticObjectDetection3DDataset(cfg)
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


__all__ = ["DataConfig", "SyntheticObjectDetection3DDataset", "get_dataloaders"]
