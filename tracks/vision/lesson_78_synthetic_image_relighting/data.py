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
    noise_std: float = 0.01


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
    if float(cfg.noise_std) < 0.0:
        raise ValueError("noise_std must be >= 0")


def _base_image(image_size: int, generator: torch.Generator) -> torch.Tensor:
    yy, xx = torch.meshgrid(
        torch.linspace(0.0, 1.0, image_size, dtype=torch.float32),
        torch.linspace(0.0, 1.0, image_size, dtype=torch.float32),
        indexing="ij",
    )
    image = torch.zeros((3, image_size, image_size), dtype=torch.float32)
    image[0] = 0.10 + 0.40 * yy + 0.08 * xx
    image[1] = 0.12 + 0.28 * yy
    image[2] = 0.08 + 0.24 * xx

    for _ in range(4):
        h = int(torch.randint(image_size // 6, image_size // 2, (1,), generator=generator).item())
        w = int(torch.randint(image_size // 6, image_size // 2, (1,), generator=generator).item())
        top = int(torch.randint(0, image_size - h + 1, (1,), generator=generator).item())
        left = int(torch.randint(0, image_size - w + 1, (1,), generator=generator).item())
        color = torch.rand((3, 1, 1), generator=generator, dtype=torch.float32) * 0.75 + 0.15
        image[:, top : top + h, left : left + w] = color

    return image.clamp(0.0, 1.0)


def _light_map(image_size: int, generator: torch.Generator) -> torch.Tensor:
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, image_size, dtype=torch.float32),
        torch.linspace(-1.0, 1.0, image_size, dtype=torch.float32),
        indexing="ij",
    )
    cx = float(torch.empty((1,)).uniform_(-0.45, 0.45, generator=generator).item())
    cy = float(torch.empty((1,)).uniform_(-0.45, 0.45, generator=generator).item())
    sigma = float(torch.empty((1,)).uniform_(0.35, 0.75, generator=generator).item())
    angle = float(torch.empty((1,)).uniform_(0.0, 2.0 * torch.pi, generator=generator).item())
    bias = float(torch.empty((1,)).uniform_(-0.12, 0.12, generator=generator).item())

    gaussian = torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma * sigma))
    directional = torch.cos(torch.tensor(angle)) * xx + torch.sin(torch.tensor(angle)) * yy
    light = 0.55 + 0.35 * gaussian + 0.15 * directional + bias
    return light.clamp(0.05, 1.0).unsqueeze(0)


def _relight(source: torch.Tensor, light_map: torch.Tensor, noise_std: float, generator: torch.Generator) -> torch.Tensor:
    gain = 0.65 + 0.70 * light_map
    relit = source * gain
    relit = relit + 0.08 * (1.0 - source) * light_map
    if noise_std > 0.0:
        relit = relit + float(noise_std) * torch.randn(source.shape, generator=generator, dtype=source.dtype)
    return relit.clamp(0.0, 1.0)


def make_sample(cfg: DataConfig, idx: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    _validate_config(cfg)
    generator = torch.Generator().manual_seed(int(cfg.seed) * 1_000_003 + int(idx) * 97 + 17)
    source = _base_image(int(cfg.image_size), generator)
    light_map = _light_map(int(cfg.image_size), generator)
    relit = _relight(source, light_map, float(cfg.noise_std), generator)
    return source, {"relit": relit, "light_map": light_map}


class SyntheticImageRelightingDataset(Dataset):
    def __init__(self, cfg: DataConfig) -> None:
        _validate_config(cfg)
        self.cfg = cfg

    def __len__(self) -> int:
        return int(self.cfg.num_samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return make_sample(self.cfg, int(idx))


def get_dataloaders(cfg: DataConfig) -> tuple[DataLoader, DataLoader]:
    _validate_config(cfg)
    ds = SyntheticImageRelightingDataset(cfg)
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
    "SyntheticImageRelightingDataset",
    "get_dataloaders",
    "make_sample",
]
