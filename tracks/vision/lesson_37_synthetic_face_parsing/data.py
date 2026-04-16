from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dlhub.data.splits import train_val_split_indices


@dataclass(frozen=True)
class DataConfig:
    num_samples: int = 1024
    batch_size: int = 16
    image_size: int = 48
    val_fraction: float = 0.2
    seed: int = 0
    num_workers: int = 0
    num_classes: int = 6


class SyntheticFaceParsingDataset:
    """Synthetic face parsing with coarse regions: background, hair, skin, eyes, mouth."""

    def __init__(self, cfg: DataConfig) -> None:
        if int(cfg.image_size) < 32:
            raise ValueError("image_size must be >= 32")
        if int(cfg.num_classes) != 6:
            raise ValueError("This lesson expects exactly 6 classes.")
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

        cx = float(rng.uniform(0.44, 0.56) * (size - 1))
        cy = float(rng.uniform(0.46, 0.58) * (size - 1))
        radius = float(rng.uniform(0.22, 0.28) * size)

        image = np.full((size, size), 0.08, dtype=np.float32)
        mask = np.zeros((size, size), dtype=np.int64)

        face_region = ((xx - cx) ** 2 + (yy - cy) ** 2) <= radius**2
        mask[face_region] = 2
        image[face_region] = 0.72

        hair_region = (((xx - cx) ** 2 + (yy - (cy - 0.18 * radius)) ** 2) <= (radius * 1.02) ** 2) & (
            yy <= cy - 0.02 * radius
        )
        mask[hair_region] = 1
        image[hair_region] = 0.38

        eye_dx = 0.34 * radius
        eye_y = cy - 0.14 * radius
        eye_radius = max(1.5, 0.10 * radius)
        left_eye = ((xx - (cx - eye_dx)) ** 2 + (yy - eye_y) ** 2) <= eye_radius**2
        right_eye = ((xx - (cx + eye_dx)) ** 2 + (yy - eye_y) ** 2) <= eye_radius**2
        mask[left_eye] = 3
        mask[right_eye] = 4
        image[left_eye] = 0.12
        image[right_eye] = 0.12

        mouth_y = cy + 0.28 * radius
        mouth_w = max(2.0, 0.24 * radius)
        mouth_h = max(1.0, 0.08 * radius)
        mouth = (((xx - cx) / mouth_w) ** 2 + ((yy - mouth_y) / mouth_h) ** 2) <= 1.0
        mask[mouth] = 5
        image[mouth] = 0.2

        image += 0.03 * (1.0 - np.clip(np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / max(radius, 1e-6), 0.0, 1.0))
        image += rng.normal(0.0, 0.03, size=(size, size)).astype(np.float32)
        image = np.clip(image, 0.0, 1.0)

        return torch.from_numpy(image).unsqueeze(0), torch.from_numpy(mask).to(torch.long)


def get_dataloaders(cfg: DataConfig):
    from torch.utils.data import DataLoader, Subset

    dataset = SyntheticFaceParsingDataset(cfg)
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


__all__ = ["DataConfig", "SyntheticFaceParsingDataset", "get_dataloaders"]
