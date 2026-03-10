from dataclasses import dataclass

import numpy as np

from dlhub.data.splits import train_val_split_indices


@dataclass(frozen=True)
class DataConfig:
    num_samples: int = 4096
    batch_size: int = 32
    image_size: int = 64
    val_fraction: float = 0.2
    seed: int = 0
    num_workers: int = 0

    noise_std: float = 0.15
    min_rect: int = 10
    max_rect: int = 28


class ToyRectanglesSegmentation:
    """Synthetic binary segmentation: one bright rectangle per image."""

    def __init__(self, cfg: DataConfig) -> None:
        if int(cfg.image_size) < 16:
            raise ValueError("image_size must be >= 16")
        if not (0.0 < float(cfg.val_fraction) < 1.0):
            raise ValueError("val_fraction must be in (0, 1)")
        if int(cfg.min_rect) < 2 or int(cfg.max_rect) < int(cfg.min_rect):
            raise ValueError("invalid rectangle size range")

        self.cfg = cfg
        rng = np.random.default_rng(int(cfg.seed))

        s = int(cfg.image_size)
        sizes = rng.integers(
            low=int(cfg.min_rect), high=int(cfg.max_rect) + 1, size=(int(cfg.num_samples), 2)
        )
        self.h = sizes[:, 0].astype(np.int64)
        self.w = sizes[:, 1].astype(np.int64)

        self.xy = np.empty((int(cfg.num_samples), 2), dtype=np.int64)
        for i in range(int(cfg.num_samples)):
            rh = int(self.h[i])
            rw = int(self.w[i])
            top = int(rng.integers(low=0, high=max(1, s - rh)))
            left = int(rng.integers(low=0, high=max(1, s - rw)))
            self.xy[i, 0] = top
            self.xy[i, 1] = left

    def __len__(self) -> int:
        return int(self.cfg.num_samples)

    def __getitem__(self, idx: int):
        import torch

        idx = int(idx)
        s = int(self.cfg.image_size)
        rh = int(self.h[idx])
        rw = int(self.w[idx])
        top = int(self.xy[idx, 0])
        left = int(self.xy[idx, 1])

        rng = np.random.default_rng(int(self.cfg.seed) * 1_000_003 + idx)

        img = rng.normal(loc=0.0, scale=float(self.cfg.noise_std), size=(s, s)).astype(np.float32)
        img = np.clip(img, -1.0, 1.0)

        mask = np.zeros((s, s), dtype=np.float32)
        mask[top : top + rh, left : left + rw] = 1.0
        img[top : top + rh, left : left + rw] = 1.0

        x = torch.from_numpy(img).unsqueeze(0)  # (1, H, W)
        y = torch.from_numpy(mask).unsqueeze(0)  # (1, H, W)
        return x, y


def get_dataloaders(cfg: DataConfig):
    """Return `(train_loader, val_loader)` for the synthetic segmentation task."""

    from torch.utils.data import DataLoader, Subset

    ds = ToyRectanglesSegmentation(cfg)
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


__all__ = ["DataConfig", "ToyRectanglesSegmentation", "get_dataloaders"]
