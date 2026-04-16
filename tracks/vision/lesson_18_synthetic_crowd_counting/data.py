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

    min_people: int = 4
    max_people: int = 20
    noise_std: float = 0.04
    point_sigma: float = 1.6


class SyntheticCrowdDataset:
    """Synthetic grayscale scenes with point annotations rendered as density maps."""

    def __init__(self, cfg: DataConfig) -> None:
        if int(cfg.image_size) < 32:
            raise ValueError("image_size must be >= 32")
        if not (0.0 < float(cfg.val_fraction) < 1.0):
            raise ValueError("val_fraction must be in (0, 1)")
        if int(cfg.min_people) < 0:
            raise ValueError("min_people must be >= 0")
        if int(cfg.max_people) < int(cfg.min_people):
            raise ValueError("max_people must be >= min_people")
        if float(cfg.point_sigma) <= 0.0:
            raise ValueError("point_sigma must be > 0")

        self.cfg = cfg
        rng = np.random.default_rng(int(cfg.seed))
        n = int(cfg.num_samples)
        s = int(cfg.image_size)

        self.counts = rng.integers(
            int(cfg.min_people),
            int(cfg.max_people) + 1,
            size=n,
            dtype=np.int64,
        )
        self.points: list[np.ndarray] = []
        border = max(4, int(np.ceil(float(cfg.point_sigma) * 3.0)))
        low = border
        high = max(border + 1, s - border)

        for count in self.counts.tolist():
            if count <= 0:
                pts = np.zeros((0, 2), dtype=np.int64)
            else:
                pts = rng.integers(low=low, high=high, size=(int(count), 2), dtype=np.int64)
            self.points.append(pts)

    def __len__(self) -> int:
        return int(self.cfg.num_samples)

    def __getitem__(self, idx: int):
        import torch

        idx = int(idx)
        s = int(self.cfg.image_size)
        yy, xx = np.mgrid[0:s, 0:s].astype(np.float32)
        density = np.zeros((s, s), dtype=np.float32)
        image = np.zeros((s, s), dtype=np.float32)
        sigma = float(self.cfg.point_sigma)
        denom = 2.0 * sigma * sigma

        for x0, y0 in self.points[idx]:
            kernel = np.exp(-((xx - float(x0)) ** 2 + (yy - float(y0)) ** 2) / denom).astype(
                np.float32
            )
            kernel_sum = float(kernel.sum())
            if kernel_sum > 0.0:
                kernel /= kernel_sum
            density += kernel
            image += kernel * 1.75

            y1 = max(0, int(y0) - 1)
            y2 = min(s, int(y0) + 2)
            x1 = max(0, int(x0) - 1)
            x2 = min(s, int(x0) + 2)
            image[y1:y2, x1:x2] += 0.15

        image = image / max(1.0, float(image.max()))
        rng = np.random.default_rng(int(self.cfg.seed) * 1_000_003 + idx)
        noise = rng.normal(0.0, float(self.cfg.noise_std), size=(s, s)).astype(np.float32)
        background = rng.uniform(0.0, 0.08, size=(s, s)).astype(np.float32)
        image = np.clip(image + background + noise, 0.0, 1.0)

        x = torch.from_numpy(image).unsqueeze(0)
        density_map = torch.from_numpy(density).unsqueeze(0)
        count = torch.tensor(float(self.counts[idx]), dtype=torch.float32)
        return x, density_map, count


def get_dataloaders(cfg: DataConfig):
    from torch.utils.data import DataLoader, Subset

    ds = SyntheticCrowdDataset(cfg)
    train_idx, val_idx = train_val_split_indices(
        n=len(ds), val_fraction=float(cfg.val_fraction), seed=int(cfg.seed)
    )
    train_loader = DataLoader(
        Subset(ds, train_idx),
        batch_size=int(cfg.batch_size),
        shuffle=True,
        num_workers=int(cfg.num_workers),
    )
    val_loader = DataLoader(
        Subset(ds, val_idx),
        batch_size=int(cfg.batch_size),
        shuffle=False,
        num_workers=int(cfg.num_workers),
    )
    return train_loader, val_loader


__all__ = ["DataConfig", "SyntheticCrowdDataset", "get_dataloaders"]
