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


def _draw_rectangle(mask: torch.Tensor, *, top: int, left: int, height: int, width: int) -> None:
    mask[:, top : top + height, left : left + width] = 1.0


def _draw_circle(mask: torch.Tensor, *, cy: int, cx: int, radius: int) -> None:
    h, w = mask.shape[-2:]
    yy = torch.arange(h, dtype=torch.float32).view(h, 1)
    xx = torch.arange(w, dtype=torch.float32).view(1, w)
    circle = ((yy - float(cy)) ** 2 + (xx - float(cx)) ** 2) <= float(radius * radius)
    mask[:, circle] = 1.0


def _compute_edges(mask: torch.Tensor) -> torch.Tensor:
    dx = (mask[:, :, 1:] - mask[:, :, :-1]).abs()
    dy = (mask[:, 1:, :] - mask[:, :-1, :]).abs()
    edge = torch.zeros_like(mask)
    edge[:, :, 1:] = torch.maximum(edge[:, :, 1:], dx)
    edge[:, :, :-1] = torch.maximum(edge[:, :, :-1], dx)
    edge[:, 1:, :] = torch.maximum(edge[:, 1:, :], dy)
    edge[:, :-1, :] = torch.maximum(edge[:, :-1, :], dy)
    return edge.clamp(0.0, 1.0)


class SyntheticEdgeDetectionDataset(Dataset):
    def __init__(self, cfg: DataConfig) -> None:
        _validate_config(cfg)
        self.cfg = cfg

    def __len__(self) -> int:
        return int(self.cfg.num_samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        size = int(self.cfg.image_size)
        gen = torch.Generator().manual_seed(int(self.cfg.seed) * 917 + int(idx) * 29)
        image = torch.rand((1, size, size), generator=gen, dtype=torch.float32) * 0.08
        mask = torch.zeros((1, size, size), dtype=torch.float32)

        shape_count = int(torch.randint(2, 5, (1,), generator=gen).item())
        for shape_idx in range(shape_count):
            if shape_idx % 2 == 0:
                height = int(torch.randint(size // 6, size // 3 + 1, (1,), generator=gen).item())
                width = int(torch.randint(size // 6, size // 3 + 1, (1,), generator=gen).item())
                top = int(torch.randint(0, size - height + 1, (1,), generator=gen).item())
                left = int(torch.randint(0, size - width + 1, (1,), generator=gen).item())
                _draw_rectangle(mask, top=top, left=left, height=height, width=width)
            else:
                radius = int(torch.randint(size // 8, size // 5 + 1, (1,), generator=gen).item())
                cy = int(torch.randint(radius, size - radius + 1, (1,), generator=gen).item())
                cx = int(torch.randint(radius, size - radius + 1, (1,), generator=gen).item())
                _draw_circle(mask, cy=cy, cx=cx, radius=radius)

        image = (image + 0.85 * mask).clamp(0.0, 1.0)
        target = _compute_edges(mask)
        return image, target


def get_dataloaders(cfg: DataConfig) -> tuple[DataLoader, DataLoader]:
    _validate_config(cfg)
    ds = SyntheticEdgeDetectionDataset(cfg)
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


__all__ = ["DataConfig", "SyntheticEdgeDetectionDataset", "get_dataloaders"]
