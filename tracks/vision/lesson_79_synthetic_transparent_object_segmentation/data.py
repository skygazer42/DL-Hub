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

    in_channels: int = 3
    alpha_min: float = 0.2
    alpha_max: float = 0.8


def _validate_config(cfg: DataConfig) -> None:
    if int(cfg.num_samples) <= 0:
        raise ValueError("num_samples must be > 0")
    if int(cfg.batch_size) <= 0:
        raise ValueError("batch_size must be > 0")
    if int(cfg.image_size) < 16:
        raise ValueError("image_size must be >= 16")
    if not (0.0 < float(cfg.val_fraction) < 1.0):
        raise ValueError("val_fraction must be in (0, 1)")
    if int(cfg.in_channels) != 3:
        raise ValueError("in_channels must be 3 for v1")
    if not (0.0 <= float(cfg.alpha_min) <= float(cfg.alpha_max) <= 1.0):
        raise ValueError("alpha range must satisfy 0 <= alpha_min <= alpha_max <= 1")


def _make_background(size: int, generator: torch.Generator) -> torch.Tensor:
    yy = torch.linspace(0.0, 1.0, size, dtype=torch.float32).view(size, 1)
    xx = torch.linspace(0.0, 1.0, size, dtype=torch.float32).view(1, size)
    bg = torch.zeros((3, size, size), dtype=torch.float32)
    bg[0] = 0.08 + 0.35 * yy + 0.12 * xx
    bg[1] = 0.14 + 0.45 * xx
    bg[2] = 0.10 + 0.30 * yy + 0.18 * xx
    bg += 0.02 * torch.rand((3, size, size), generator=generator, dtype=torch.float32)
    return bg.clamp(0.0, 1.0)


def _sample_ellipse_mask(size: int, generator: torch.Generator) -> torch.Tensor:
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, size, dtype=torch.float32),
        torch.linspace(-1.0, 1.0, size, dtype=torch.float32),
        indexing="ij",
    )
    cx = float(torch.empty((), dtype=torch.float32).uniform_(-0.25, 0.25, generator=generator).item())
    cy = float(torch.empty((), dtype=torch.float32).uniform_(-0.25, 0.25, generator=generator).item())
    rx = float(torch.empty((), dtype=torch.float32).uniform_(0.20, 0.55, generator=generator).item())
    ry = float(torch.empty((), dtype=torch.float32).uniform_(0.20, 0.55, generator=generator).item())
    level = ((xx - cx) / rx).pow(2) + ((yy - cy) / ry).pow(2)
    mask = (level <= 1.0).to(torch.float32).unsqueeze(0)
    return mask


def _mask_boundary(mask: torch.Tensor) -> torch.Tensor:
    avg = torch.nn.functional.avg_pool2d(mask.unsqueeze(0), kernel_size=3, stride=1, padding=1).squeeze(0)
    boundary = torch.abs(mask - avg)
    return boundary.clamp(0.0, 1.0)


def _synthesize_sample(cfg: DataConfig, idx: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    seed = int(cfg.seed) * 1_000_003 + int(idx) * 97
    generator = torch.Generator().manual_seed(seed)
    size = int(cfg.image_size)

    background = _make_background(size, generator)
    mask = _sample_ellipse_mask(size, generator)
    boundary = _mask_boundary(mask)
    alpha_value = float(torch.empty((), dtype=torch.float32).uniform_(cfg.alpha_min, cfg.alpha_max, generator=generator).item())
    alpha = mask * alpha_value

    tint = torch.rand((3, 1, 1), generator=generator, dtype=torch.float32) * 0.5 + 0.5
    image = background * (1.0 - alpha) + tint * alpha
    image += 0.01 * torch.rand((3, size, size), generator=generator, dtype=torch.float32)
    image = image.clamp(0.0, 1.0)
    return image, {"mask": mask, "alpha": alpha, "boundary": boundary}


class SyntheticTransparentObjectSegmentationDataset(Dataset):
    def __init__(self, cfg: DataConfig) -> None:
        _validate_config(cfg)
        self.cfg = cfg

    def __len__(self) -> int:
        return int(self.cfg.num_samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return _synthesize_sample(self.cfg, int(idx))


def get_dataloaders(cfg: DataConfig) -> tuple[DataLoader, DataLoader]:
    _validate_config(cfg)
    ds = SyntheticTransparentObjectSegmentationDataset(cfg)
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
    "SyntheticTransparentObjectSegmentationDataset",
    "get_dataloaders",
]
