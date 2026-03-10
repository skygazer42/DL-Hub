from dataclasses import dataclass

import numpy as np

from dlhub.data.splits import train_val_split_indices


@dataclass(frozen=True)
class DataConfig:
    num_samples: int = 4096
    batch_size: int = 64
    image_size: int = 64
    val_fraction: float = 0.2
    seed: int = 0
    num_workers: int = 0

    noise_std: float = 0.15
    min_square: int = 8
    max_square: int = 20
    num_classes: int = 4  # quadrant classification


def _quadrant_bounds(image_size: int, quadrant: int) -> tuple[int, int, int, int]:
    h = int(image_size)
    w = int(image_size)
    mid_y = h // 2
    mid_x = w // 2

    if quadrant == 0:  # top-left
        return 0, mid_y, 0, mid_x
    if quadrant == 1:  # top-right
        return 0, mid_y, mid_x, w
    if quadrant == 2:  # bottom-left
        return mid_y, h, 0, mid_x
    if quadrant == 3:  # bottom-right
        return mid_y, h, mid_x, w
    raise ValueError("quadrant must be in {0,1,2,3}")


class ToyQuadrantSquares:
    def __init__(self, cfg: DataConfig) -> None:
        if int(cfg.image_size) < 16:
            raise ValueError("image_size must be >= 16")
        if not (0.0 < float(cfg.val_fraction) < 1.0):
            raise ValueError("val_fraction must be in (0, 1)")
        if int(cfg.num_classes) != 4:
            raise ValueError("This toy dataset currently uses 4 classes (quadrants).")
        if int(cfg.min_square) < 2 or int(cfg.max_square) < int(cfg.min_square):
            raise ValueError("invalid square size range")

        self.cfg = cfg
        rng = np.random.default_rng(int(cfg.seed))

        self.labels = rng.integers(low=0, high=4, size=(int(cfg.num_samples),), dtype=np.int64)
        self.sizes = rng.integers(
            low=int(cfg.min_square),
            high=int(cfg.max_square) + 1,
            size=(int(cfg.num_samples),),
            dtype=np.int64,
        )

        # Pre-sample square positions per item to keep dataset deterministic.
        self.xy = np.empty((int(cfg.num_samples), 2), dtype=np.int64)
        for i in range(int(cfg.num_samples)):
            q = int(self.labels[i])
            size = int(self.sizes[i])
            y0, y1, x0, x1 = _quadrant_bounds(int(cfg.image_size), q)
            # Ensure square fits inside the quadrant.
            y1 = max(y0 + size + 1, y1)
            x1 = max(x0 + size + 1, x1)
            top = int(rng.integers(low=y0, high=max(y0 + 1, y1 - size)))
            left = int(rng.integers(low=x0, high=max(x0 + 1, x1 - size)))
            self.xy[i, 0] = top
            self.xy[i, 1] = left

    def __len__(self) -> int:
        return int(self.cfg.num_samples)

    def __getitem__(self, idx: int):
        import torch

        idx = int(idx)
        s = int(self.cfg.image_size)
        label = int(self.labels[idx])
        size = int(self.sizes[idx])
        top = int(self.xy[idx, 0])
        left = int(self.xy[idx, 1])

        # Per-sample RNG: deterministic across dataloader workers.
        rng = np.random.default_rng(int(self.cfg.seed) * 1_000_003 + idx)

        img = rng.normal(loc=0.0, scale=float(self.cfg.noise_std), size=(s, s)).astype(np.float32)
        img = np.clip(img, -1.0, 1.0)
        img[top : top + size, left : left + size] = 1.0

        x = torch.from_numpy(img).unsqueeze(0)  # (1, H, W)
        y = torch.tensor(label, dtype=torch.long)
        return x, y


def get_dataloaders(cfg: DataConfig):
    """Return `(train_loader, val_loader)` for the toy quadrant squares task."""

    from torch.utils.data import DataLoader, Subset

    ds = ToyQuadrantSquares(cfg)
    train_idx, val_idx = train_val_split_indices(
        n=len(ds), val_fraction=float(cfg.val_fraction), seed=int(cfg.seed)
    )
    train_ds = Subset(ds, train_idx)
    val_ds = Subset(ds, val_idx)

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg.batch_size),
        shuffle=True,
        num_workers=int(cfg.num_workers),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg.batch_size),
        shuffle=False,
        num_workers=int(cfg.num_workers),
    )
    return train_loader, val_loader


__all__ = ["DataConfig", "ToyQuadrantSquares", "get_dataloaders"]
