from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from dlhub.data.splits import train_val_split_indices


@dataclass(frozen=True)
class DataConfig:
    num_samples: int = 1024
    batch_size: int = 16
    image_size: int = 32
    val_fraction: float = 0.2
    seed: int = 0
    num_workers: int = 0

    in_channels: int = 1
    min_object_size: int = 6
    max_object_size: int = 12
    camouflage_delta: float = 0.08
    noise_std: float = 0.04


def _validate_config(cfg: DataConfig) -> None:
    if int(cfg.num_samples) <= 0:
        raise ValueError("num_samples must be > 0")
    if int(cfg.batch_size) <= 0:
        raise ValueError("batch_size must be > 0")
    if int(cfg.image_size) < 16:
        raise ValueError("image_size must be >= 16")
    if not (0.0 < float(cfg.val_fraction) < 1.0):
        raise ValueError("val_fraction must be in (0, 1)")
    if int(cfg.in_channels) != 1:
        raise ValueError("in_channels must be 1 for v1")
    if int(cfg.min_object_size) < 4 or int(cfg.max_object_size) < int(cfg.min_object_size):
        raise ValueError("invalid object size range")
    if float(cfg.camouflage_delta) <= 0.0:
        raise ValueError("camouflage_delta must be > 0")


def make_camouflaged_sample(cfg: DataConfig, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
    _validate_config(cfg)
    size = int(cfg.image_size)
    generator = torch.Generator().manual_seed(int(cfg.seed) * 1_000_003 + int(idx))

    yy = torch.linspace(0.0, 1.0, size, dtype=torch.float32).view(size, 1)
    xx = torch.linspace(0.0, 1.0, size, dtype=torch.float32).view(1, size)
    phase = float(torch.rand((), generator=generator).item()) * 3.14159
    background = 0.45 + 0.12 * torch.sin(6.0 * xx + phase) * torch.cos(5.0 * yy - phase)
    background += torch.randn((size, size), generator=generator, dtype=torch.float32) * float(cfg.noise_std)

    obj_h = int(torch.randint(int(cfg.min_object_size), int(cfg.max_object_size) + 1, (1,), generator=generator).item())
    obj_w = int(torch.randint(int(cfg.min_object_size), int(cfg.max_object_size) + 1, (1,), generator=generator).item())
    top = int(torch.randint(1, size - obj_h - 1, (1,), generator=generator).item())
    left = int(torch.randint(1, size - obj_w - 1, (1,), generator=generator).item())

    mask = torch.zeros((size, size), dtype=torch.float32)
    mask[top : top + obj_h, left : left + obj_w] = 1.0

    local_patch = background[top : top + obj_h, left : left + obj_w]
    delta = float(cfg.camouflage_delta)
    sign = -1.0 if bool(torch.randint(0, 2, (1,), generator=generator).item()) else 1.0
    object_texture = local_patch + sign * delta
    object_texture += 0.02 * torch.sin(torch.linspace(0.0, 3.14159, obj_w).view(1, obj_w))
    background[top : top + obj_h, left : left + obj_w] = object_texture

    boundary = torch.nn.functional.avg_pool2d(mask.unsqueeze(0).unsqueeze(0), kernel_size=3, stride=1, padding=1)
    boundary = (boundary.squeeze(0).squeeze(0) - mask).abs()
    image = (background + 0.03 * boundary).clamp(0.0, 1.0).unsqueeze(0)
    return image, mask.unsqueeze(0)


class SyntheticCamouflagedObjectDataset(Dataset):
    def __init__(self, cfg: DataConfig) -> None:
        _validate_config(cfg)
        self.cfg = cfg

    def __len__(self) -> int:
        return int(self.cfg.num_samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return make_camouflaged_sample(self.cfg, int(idx))


def get_dataloaders(cfg: DataConfig) -> tuple[DataLoader, DataLoader]:
    _validate_config(cfg)
    ds = SyntheticCamouflagedObjectDataset(cfg)
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
        pin_memory=False,
    )
    val_loader = DataLoader(
        Subset(ds, val_idx),
        batch_size=int(cfg.batch_size),
        shuffle=False,
        num_workers=int(cfg.num_workers),
        pin_memory=False,
    )
    return train_loader, val_loader


__all__ = [
    "DataConfig",
    "SyntheticCamouflagedObjectDataset",
    "get_dataloaders",
    "make_camouflaged_sample",
]

