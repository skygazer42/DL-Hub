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
    max_objects: int = 3
    noise_std: float = 0.01


class SyntheticLayoutGenerationDataset(Dataset):
    """Synthetic object layout generation from sparse condition hints."""

    def __init__(self, cfg: DataConfig) -> None:
        self.cfg = cfg
        if int(cfg.image_size) < 16:
            raise ValueError("image_size must be >= 16")
        if int(cfg.in_channels) < 1:
            raise ValueError("in_channels must be >= 1")
        if int(cfg.max_objects) < 1:
            raise ValueError("max_objects must be >= 1")
        if not (0.0 < float(cfg.val_fraction) < 1.0):
            raise ValueError("val_fraction must be in (0, 1)")
        if float(cfg.noise_std) < 0.0:
            raise ValueError("noise_std must be >= 0")

    def __len__(self) -> int:
        return int(self.cfg.num_samples)

    def _generator(self, idx: int) -> torch.Generator:
        seed = int(self.cfg.seed) * 1_000_003 + int(idx) * 97 + 82
        return torch.Generator().manual_seed(seed)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        cfg = self.cfg
        gen = self._generator(int(idx))
        size = int(cfg.image_size)
        channels = int(cfg.in_channels)

        condition = torch.full((channels, size, size), 0.03, dtype=torch.float32)
        layout = torch.zeros((channels, size, size), dtype=torch.float32)

        num_objects = int(torch.randint(1, int(cfg.max_objects) + 1, (1,), generator=gen).item())
        for _ in range(num_objects):
            channel = int(torch.randint(0, channels, (1,), generator=gen).item())
            width = int(torch.randint(max(4, size // 8), max(5, size // 3), (1,), generator=gen).item())
            height = int(torch.randint(max(4, size // 8), max(5, size // 3), (1,), generator=gen).item())
            x0 = int(torch.randint(1, max(2, size - width - 1), (1,), generator=gen).item())
            y0 = int(torch.randint(1, max(2, size - height - 1), (1,), generator=gen).item())
            x1 = min(size, x0 + width)
            y1 = min(size, y0 + height)

            intensity = float(torch.empty((1,), dtype=torch.float32).uniform_(0.55, 1.0, generator=gen).item())
            layout[channel, y0:y1, x0:x1] = torch.maximum(
                layout[channel, y0:y1, x0:x1],
                torch.full((y1 - y0, x1 - x0), intensity, dtype=torch.float32),
            )

            # Sparse hints: box border + center point.
            condition[channel, y0, x0:x1] = 1.0
            condition[channel, y1 - 1, x0:x1] = 1.0
            condition[channel, y0:y1, x0] = 1.0
            condition[channel, y0:y1, x1 - 1] = 1.0
            cy = (y0 + y1 - 1) // 2
            cx = (x0 + x1 - 1) // 2
            condition[channel, cy, cx] = 1.0

        if float(cfg.noise_std) > 0.0:
            condition = condition + torch.randn(condition.shape, generator=gen, dtype=torch.float32) * float(
                cfg.noise_std
            )
        condition = condition.clamp(0.0, 1.0)
        occupancy = (layout.max(dim=0, keepdim=True).values > 0.0).to(torch.float32)
        return condition, {"layout": layout.clamp(0.0, 1.0), "occupancy": occupancy}


def get_dataloaders(cfg: DataConfig) -> tuple[DataLoader, DataLoader]:
    dataset = SyntheticLayoutGenerationDataset(cfg)
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


__all__ = ["DataConfig", "SyntheticLayoutGenerationDataset", "get_dataloaders"]
