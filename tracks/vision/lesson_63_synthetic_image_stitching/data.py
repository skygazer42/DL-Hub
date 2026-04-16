from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from dlhub.data.splits import train_val_split_indices


@dataclass(frozen=True)
class DataConfig:
    num_samples: int = 1024
    batch_size: int = 16
    image_size: int = 48
    val_fraction: float = 0.2
    seed: int = 0
    num_workers: int = 0

    in_channels: int = 3
    min_shapes: int = 3
    max_shapes: int = 6
    overlap_width: int = 12


def _validate_config(cfg: DataConfig) -> None:
    if int(cfg.num_samples) <= 0:
        raise ValueError("num_samples must be > 0")
    if int(cfg.batch_size) <= 0:
        raise ValueError("batch_size must be > 0")
    if int(cfg.image_size) < 24:
        raise ValueError("image_size must be >= 24")
    if not (0.0 < float(cfg.val_fraction) < 1.0):
        raise ValueError("val_fraction must be in (0,1)")
    if int(cfg.in_channels) != 3:
        raise ValueError("in_channels must be 3 for v1")
    if int(cfg.min_shapes) <= 0 or int(cfg.max_shapes) < int(cfg.min_shapes):
        raise ValueError("invalid shape range")
    if int(cfg.overlap_width) < 4 or int(cfg.overlap_width) >= int(cfg.image_size):
        raise ValueError("overlap_width must be in [4, image_size)")


def _seed_for(cfg: DataConfig, idx: int, *, salt: int = 0) -> int:
    return int(cfg.seed) * 1_000_003 + int(idx) * 97 + int(salt)


def _sample_color(generator: torch.Generator) -> torch.Tensor:
    return torch.rand((3,), generator=generator, dtype=torch.float32) * 0.70 + 0.20


def _draw_rectangle(
    image: torch.Tensor,
    *,
    top: int,
    left: int,
    height: int,
    width: int,
    color: torch.Tensor,
) -> None:
    image[:, top : top + height, left : left + width] = color.view(3, 1, 1)


def _draw_circle(
    image: torch.Tensor,
    *,
    cy: int,
    cx: int,
    radius: int,
    color: torch.Tensor,
) -> None:
    height, width = image.shape[-2:]
    yy = torch.arange(height, dtype=torch.float32).view(height, 1)
    xx = torch.arange(width, dtype=torch.float32).view(1, width)
    mask = ((yy - float(cy)) ** 2 + (xx - float(cx)) ** 2) <= float(radius * radius)
    image[:, mask] = color.view(3, 1)


def make_panorama(cfg: DataConfig, idx: int) -> torch.Tensor:
    _validate_config(cfg)
    size = int(cfg.image_size)
    generator = torch.Generator().manual_seed(_seed_for(cfg, idx))

    yy = torch.linspace(0.0, 1.0, size, dtype=torch.float32).view(size, 1)
    xx = torch.linspace(0.0, 1.0, size, dtype=torch.float32).view(1, size)
    panorama = torch.zeros((3, size, size), dtype=torch.float32)
    panorama[0] = 0.10 + 0.24 * yy + 0.14 * xx
    panorama[1] = 0.08 + 0.18 * yy + 0.20 * (1.0 - xx)
    panorama[2] = 0.14 + 0.12 * yy + 0.22 * xx

    num_shapes = int(
        torch.randint(
            low=int(cfg.min_shapes),
            high=int(cfg.max_shapes) + 1,
            size=(1,),
            generator=generator,
        ).item()
    )
    min_size = max(4, size // 9)
    max_size = max(min_size + 1, size // 3)

    for shape_idx in range(num_shapes):
        color = _sample_color(generator)
        if shape_idx % 2 == 0:
            rect_h = int(torch.randint(min_size, max_size + 1, (1,), generator=generator).item())
            rect_w = int(torch.randint(min_size, max_size + 1, (1,), generator=generator).item())
            top = int(torch.randint(0, size - rect_h + 1, (1,), generator=generator).item())
            left = int(torch.randint(0, size - rect_w + 1, (1,), generator=generator).item())
            _draw_rectangle(panorama, top=top, left=left, height=rect_h, width=rect_w, color=color)
            continue

        radius = int(torch.randint(min_size // 2, max_size // 2 + 1, (1,), generator=generator).item())
        cy = int(torch.randint(radius, size - radius + 1, (1,), generator=generator).item())
        cx = int(torch.randint(radius, size - radius + 1, (1,), generator=generator).item())
        _draw_circle(panorama, cy=cy, cx=cx, radius=radius, color=color)

    return panorama.clamp(0.0, 1.0)


def synthesize_view_pair(
    panorama: torch.Tensor,
    *,
    cfg: DataConfig,
    idx: int,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    if panorama.ndim != 3:
        raise ValueError("panorama must have shape (C,H,W)")
    _validate_config(cfg)

    width = int(panorama.shape[-1])
    overlap = int(cfg.overlap_width)
    left_end = min(width - 1, width // 2 + overlap // 2)
    right_start = max(1, left_end - overlap)

    generator = torch.Generator().manual_seed(_seed_for(cfg, idx, salt=17))
    left_gain = float(torch.rand((), generator=generator).item() * 0.16 + 0.92)
    right_gain = float(torch.rand((), generator=generator).item() * 0.16 + 0.92)
    left_bias = float(torch.rand((), generator=generator).item() * 0.04 - 0.02)
    right_bias = float(torch.rand((), generator=generator).item() * 0.04 - 0.02)

    left_view = torch.zeros_like(panorama)
    right_view = torch.zeros_like(panorama)
    left_view[:, :, :left_end] = (panorama[:, :, :left_end] * left_gain + left_bias).clamp(0.0, 1.0)
    right_view[:, :, right_start:] = (
        panorama[:, :, right_start:] * right_gain + right_bias
    ).clamp(0.0, 1.0)

    return {"left_view": left_view, "right_view": right_view}, panorama


class SyntheticImageStitchingDataset(Dataset):
    def __init__(self, cfg: DataConfig) -> None:
        _validate_config(cfg)
        self.cfg = cfg

    def __len__(self) -> int:
        return int(self.cfg.num_samples)

    def __getitem__(self, idx: int) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        panorama = make_panorama(self.cfg, int(idx))
        return synthesize_view_pair(panorama, cfg=self.cfg, idx=int(idx))


def get_dataloaders(cfg: DataConfig) -> tuple[DataLoader, DataLoader]:
    _validate_config(cfg)
    dataset = SyntheticImageStitchingDataset(cfg)
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
        pin_memory=False,
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=int(cfg.batch_size),
        shuffle=False,
        num_workers=int(cfg.num_workers),
        pin_memory=False,
    )
    return train_loader, val_loader


__all__ = [
    "DataConfig",
    "SyntheticImageStitchingDataset",
    "get_dataloaders",
    "make_panorama",
    "synthesize_view_pair",
]
