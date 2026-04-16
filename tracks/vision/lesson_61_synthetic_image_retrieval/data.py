from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dlhub.data.splits import train_val_split_indices


@dataclass(frozen=True)
class DataConfig:
    num_samples: int = 2048
    batch_size: int = 32
    image_size: int = 48
    val_fraction: float = 0.2
    seed: int = 0
    num_workers: int = 0
    in_channels: int = 1
    num_categories: int = 6
    noise_std: float = 0.04


def _render_category(*, category: int, variation_seed: int, size: int, noise_std: float) -> np.ndarray:
    rng = np.random.default_rng(int(variation_seed) * 1_000_003 + int(category) * 97)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    image = np.full((size, size), 0.06, dtype=np.float32)

    image += 0.08 * (xx / max(size - 1, 1)).astype(np.float32)
    image += 0.04 * (yy / max(size - 1, 1)).astype(np.float32)

    row = int(category) // 3
    col = int(category) % 3
    cx = float((0.24 + 0.26 * col + rng.uniform(-0.03, 0.03)) * (size - 1))
    cy = float((0.26 + 0.22 * row + rng.uniform(-0.03, 0.03)) * (size - 1))
    radius = float((0.11 + 0.01 * category) * size)
    circle = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * radius * radius)).astype(np.float32)
    image += (0.28 + 0.03 * row) * circle

    angle = float((category + 1) * np.pi / 7.0)
    centered_x = xx - cx
    centered_y = yy - cy
    rotated = centered_x * np.cos(angle) + centered_y * np.sin(angle)
    stripe = np.exp(-(rotated**2) / (2.0 * (1.2 + 0.2 * col) ** 2)).astype(np.float32)
    image += (0.16 + 0.02 * col) * stripe

    patch_size = int(max(4, 0.12 * size))
    patch_top = int(np.clip(cy + (-0.24 + 0.08 * row) * size, 0, size - patch_size))
    patch_left = int(np.clip(cx + (-0.22 + 0.10 * col) * size, 0, size - patch_size))
    image[patch_top : patch_top + patch_size, patch_left : patch_left + patch_size] += 0.22 + 0.03 * category

    ring_radius = float((0.22 + 0.01 * category) * size)
    dist = np.sqrt((xx - (size * 0.52)) ** 2 + (yy - (size * 0.54)) ** 2)
    ring = np.exp(-((dist - ring_radius) ** 2) / (2.0 * 1.8 * 1.8)).astype(np.float32)
    image += (0.05 + 0.01 * (category % 2)) * ring

    image += rng.normal(0.0, float(noise_std), size=(size, size)).astype(np.float32)
    return np.clip(image, 0.0, 1.0)


class SyntheticImageRetrievalDataset:
    """Deterministic synthetic gallery for embedding-based image retrieval."""

    def __init__(self, cfg: DataConfig) -> None:
        if int(cfg.image_size) < 32:
            raise ValueError("image_size must be >= 32")
        if int(cfg.in_channels) != 1:
            raise ValueError("This lesson expects grayscale inputs.")
        if int(cfg.num_categories) < 4:
            raise ValueError("num_categories must be >= 4")
        if not (0.0 < float(cfg.val_fraction) < 1.0):
            raise ValueError("val_fraction must be in (0, 1)")
        self.cfg = cfg

    def __len__(self) -> int:
        return int(self.cfg.num_samples)

    def __getitem__(self, idx: int):
        import torch

        category = int((int(idx) + int(self.cfg.seed)) % int(self.cfg.num_categories))
        image = _render_category(
            category=category,
            variation_seed=int(self.cfg.seed) * 10_000 + int(idx),
            size=int(self.cfg.image_size),
            noise_std=float(self.cfg.noise_std),
        )
        return torch.from_numpy(image).unsqueeze(0), category


def get_dataloaders(cfg: DataConfig):
    import torch
    from torch.utils.data import DataLoader, Subset

    dataset = SyntheticImageRetrievalDataset(cfg)
    train_idx, val_idx = train_val_split_indices(
        n=len(dataset),
        val_fraction=float(cfg.val_fraction),
        seed=int(cfg.seed),
    )

    def _collate(batch):
        images = torch.stack([item[0] for item in batch], dim=0)
        labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
        return images, labels

    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=int(cfg.batch_size),
        shuffle=True,
        num_workers=int(cfg.num_workers),
        collate_fn=_collate,
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=int(cfg.batch_size),
        shuffle=False,
        num_workers=int(cfg.num_workers),
        collate_fn=_collate,
    )
    return train_loader, val_loader


__all__ = ["DataConfig", "SyntheticImageRetrievalDataset", "get_dataloaders"]
