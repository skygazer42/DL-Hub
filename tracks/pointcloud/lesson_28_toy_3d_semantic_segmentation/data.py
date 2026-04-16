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


def _semantic_labels(points: torch.Tensor, *, num_classes: int) -> torch.Tensor:
    theta = torch.atan2(points[:, 1], points[:, 0]) + math.pi  # [0, 2pi]
    sector = torch.floor(theta / (2.0 * math.pi) * float(num_classes)).to(torch.long)
    return torch.clamp(sector, min=0, max=int(num_classes) - 1)


@dataclass(frozen=True)
class DataConfig:
    num_samples: int = 256
    num_points: int = 96
    num_classes: int = 4
    batch_size: int = 16
    val_fraction: float = 0.2
    seed: int = 0
    num_workers: int = 0
    jitter_std: float = 0.01


class ToySemanticSegmentation3DDataset(Dataset):
    def __init__(self, cfg: DataConfig) -> None:
        self.cfg = cfg
        if int(cfg.num_points) < 16:
            raise ValueError("num_points must be >= 16")
        if int(cfg.num_classes) < 2:
            raise ValueError("num_classes must be >= 2")
        if not (0.0 < float(cfg.val_fraction) < 1.0):
            raise ValueError("val_fraction must be in (0, 1)")
        if float(cfg.jitter_std) < 0.0:
            raise ValueError("jitter_std must be >= 0")

    def __len__(self) -> int:
        return int(self.cfg.num_samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample_idx = int(idx)
        cfg = self.cfg
        g = torch.Generator().manual_seed(int(cfg.seed) * 1_000_003 + sample_idx)

        if bool(torch.rand((), generator=g).item() < 0.5):
            points = _sample_cube_surface(num_points=int(cfg.num_points), g=g)
        else:
            points = _sample_sphere_surface(num_points=int(cfg.num_points), g=g)

        if float(cfg.jitter_std) > 0.0:
            points = points + float(cfg.jitter_std) * torch.randn(
                points.shape, generator=g, dtype=points.dtype
            )

        labels = _semantic_labels(points, num_classes=int(cfg.num_classes))
        return points.to(torch.float32), labels.to(torch.long)


def get_dataloaders(cfg: DataConfig) -> tuple[DataLoader, DataLoader]:
    dataset = ToySemanticSegmentation3DDataset(cfg)
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


__all__ = ["DataConfig", "ToySemanticSegmentation3DDataset", "get_dataloaders"]
