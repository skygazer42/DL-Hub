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

    noise_std: float = 0.10
    dot_sigma: float = 1.5


class SyntheticKeypointDots:
    """Synthetic grayscale images with a single bright dot.

    Target is `(x_norm, y_norm)` normalized to [0, 1].
    """

    def __init__(self, cfg: DataConfig) -> None:
        if int(cfg.image_size) < 16:
            raise ValueError("image_size must be >= 16")
        if not (0.0 < float(cfg.val_fraction) < 1.0):
            raise ValueError("val_fraction must be in (0, 1)")
        if float(cfg.dot_sigma) <= 0:
            raise ValueError("dot_sigma must be > 0")

        self.cfg = cfg
        rng = np.random.default_rng(int(cfg.seed))

        s = int(cfg.image_size)
        self.xy = rng.integers(
            low=0, high=s, size=(int(cfg.num_samples), 2), dtype=np.int64
        )  # (x, y)

    def __len__(self) -> int:
        return int(self.cfg.num_samples)

    def __getitem__(self, idx: int):
        import torch

        idx = int(idx)
        s = int(self.cfg.image_size)
        x0 = int(self.xy[idx, 0])
        y0 = int(self.xy[idx, 1])

        # Per-sample RNG for deterministic data across dataloader workers.
        rng = np.random.default_rng(int(self.cfg.seed) * 1_000_003 + idx)

        yy, xx = np.mgrid[0:s, 0:s].astype(np.float32)
        sigma = float(self.cfg.dot_sigma)
        dot = np.exp(
            -((xx - float(x0)) ** 2 + (yy - float(y0)) ** 2) / (2.0 * sigma * sigma)
        ).astype(np.float32)
        dot = dot / max(1e-6, float(dot.max()))

        noise = rng.normal(loc=0.0, scale=float(self.cfg.noise_std), size=(s, s)).astype(np.float32)
        img = np.clip(dot + noise, -1.0, 1.0)

        x = torch.from_numpy(img).unsqueeze(0)  # (1, H, W)
        target = torch.tensor([x0 / float(s - 1), y0 / float(s - 1)], dtype=torch.float32)
        return x, target


def get_dataloaders(cfg: DataConfig):
    """Return `(train_loader, val_loader)` for the synthetic keypoint regression task."""

    from torch.utils.data import DataLoader, Subset

    ds = SyntheticKeypointDots(cfg)
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


__all__ = ["DataConfig", "SyntheticKeypointDots", "get_dataloaders"]
