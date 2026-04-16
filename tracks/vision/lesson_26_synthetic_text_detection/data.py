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
    empty_fraction: float = 0.25


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
    if not (0.0 <= float(cfg.empty_fraction) < 1.0):
        raise ValueError("empty_fraction must be in [0, 1)")


def _draw_text_patch(image: torch.Tensor, *, top: int, left: int, height: int, width: int) -> None:
    yy = torch.arange(height, dtype=torch.float32).view(height, 1)
    xx = torch.arange(width, dtype=torch.float32).view(1, width)
    stripe = ((yy % 3 == 0) | (xx % 4 == 0)).to(torch.float32)
    image[:, top : top + height, left : left + width] = torch.stack(
        [
            0.85 * stripe + 0.10,
            0.90 * stripe + 0.05,
            0.80 * stripe + 0.12,
        ],
        dim=0,
    )


class SyntheticTextDetectionDataset(Dataset):
    def __init__(self, cfg: DataConfig) -> None:
        _validate_config(cfg)
        self.cfg = cfg

    def __len__(self) -> int:
        return int(self.cfg.num_samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        size = int(self.cfg.image_size)
        gen = torch.Generator().manual_seed(int(self.cfg.seed) * 1009 + int(idx) * 37)

        image = torch.rand((3, size, size), generator=gen, dtype=torch.float32) * 0.18 + 0.05
        image[1] += torch.linspace(0.0, 0.25, size, dtype=torch.float32).view(size, 1)
        image = image.clamp(0.0, 1.0)

        is_empty = bool(torch.rand((), generator=gen).item() < float(self.cfg.empty_fraction))
        if is_empty:
            bbox = torch.zeros((4,), dtype=torch.float32)
            score = torch.tensor(0.0, dtype=torch.float32)
            return image, {"bbox": bbox, "score": score}

        height = int(torch.randint(size // 6, size // 3 + 1, (1,), generator=gen).item())
        width = int(torch.randint(size // 3, int(size * 0.75) + 1, (1,), generator=gen).item())
        top = int(torch.randint(0, size - height + 1, (1,), generator=gen).item())
        left = int(torch.randint(0, size - width + 1, (1,), generator=gen).item())
        _draw_text_patch(image, top=top, left=left, height=height, width=width)

        bbox = torch.tensor(
            [
                float(left) / float(size),
                float(top) / float(size),
                float(left + width) / float(size),
                float(top + height) / float(size),
            ],
            dtype=torch.float32,
        )
        score = torch.tensor(1.0, dtype=torch.float32)
        return image.clamp(0.0, 1.0), {"bbox": bbox, "score": score}


def get_dataloaders(cfg: DataConfig) -> tuple[DataLoader, DataLoader]:
    _validate_config(cfg)
    ds = SyntheticTextDetectionDataset(cfg)
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


__all__ = ["DataConfig", "SyntheticTextDetectionDataset", "get_dataloaders"]
