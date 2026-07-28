from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dlhub.data.splits import train_val_split_indices


@dataclass(frozen=True)
class DataConfig:
    num_samples: int = 1024
    batch_size: int = 16
    image_size: int = 64
    val_fraction: float = 0.2
    seed: int = 0
    num_workers: int = 0
    in_channels: int = 1
    num_classes: int = 3


class SyntheticMedicalSegmentationDataset:
    """Synthetic medical slices with background, tissue and lesion labels."""

    def __init__(self, cfg: DataConfig) -> None:
        if int(cfg.image_size) < 48:
            raise ValueError("image_size must be >= 48")
        if int(cfg.in_channels) != 1:
            raise ValueError("This lesson expects in_channels=1")
        if int(cfg.num_classes) != 3:
            raise ValueError("This lesson expects num_classes=3")
        if not (0.0 < float(cfg.val_fraction) < 1.0):
            raise ValueError("val_fraction must be in (0, 1)")
        self.cfg = cfg

    def __len__(self) -> int:
        return int(self.cfg.num_samples)

    def __getitem__(self, idx: int):
        import torch

        size = int(self.cfg.image_size)
        rng = np.random.default_rng(int(self.cfg.seed) * 1_000_003 + int(idx))
        yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)

        center_x = float(rng.uniform(0.44, 0.56) * (size - 1))
        center_y = float(rng.uniform(0.44, 0.56) * (size - 1))
        organ_rx = float(rng.uniform(0.28, 0.36) * size)
        organ_ry = float(rng.uniform(0.24, 0.34) * size)

        mask = np.zeros((size, size), dtype=np.int64)
        organ = (((xx - center_x) / max(organ_rx, 1e-6)) ** 2 + ((yy - center_y) / max(organ_ry, 1e-6)) ** 2) <= 1.0
        mask[organ] = 1

        lesion_x = center_x + float(rng.uniform(-0.20, 0.20) * organ_rx)
        lesion_y = center_y + float(rng.uniform(-0.20, 0.20) * organ_ry)
        lesion_r = float(rng.uniform(0.10, 0.18) * min(organ_rx, organ_ry))
        lesion = (((xx - lesion_x) ** 2 + (yy - lesion_y) ** 2) <= lesion_r**2) & organ
        mask[lesion] = 2

        image = np.full((size, size), 0.08, dtype=np.float32)
        image[organ] = 0.52
        image[lesion] = 0.82

        radial = np.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2) / max(max(organ_rx, organ_ry), 1e-6)
        image = image + 0.07 * np.clip(1.0 - radial, 0.0, 1.0)
        image = image + rng.normal(0.0, 0.03, size=(size, size)).astype(np.float32)
        image = np.clip(image, 0.0, 1.0)

        return torch.from_numpy(image).unsqueeze(0), torch.from_numpy(mask).to(torch.long)


def get_dataloaders(cfg: DataConfig):
    from torch.utils.data import DataLoader, Subset

    dataset = SyntheticMedicalSegmentationDataset(cfg)
    train_idx, val_idx = train_val_split_indices(
        n=len(dataset), val_fraction=float(cfg.val_fraction), seed=int(cfg.seed)
    )
    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=int(cfg.batch_size),
        shuffle=True,
        num_workers=int(cfg.num_workers),
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=int(cfg.batch_size),
        shuffle=False,
        num_workers=int(cfg.num_workers),
    )
    return train_loader, val_loader


__all__ = ["DataConfig", "SyntheticMedicalSegmentationDataset", "get_dataloaders"]

