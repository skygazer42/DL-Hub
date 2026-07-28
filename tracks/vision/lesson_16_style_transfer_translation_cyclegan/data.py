from __future__ import annotations

from dataclasses import dataclass

import numpy as np

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

    noise_std: float = 0.15
    square_min: int = 6
    square_max: int = 12
    stripe_period: int = 6


def _make_square(rng: np.random.Generator, *, image_size: int, noise_std: float, square_min: int, square_max: int) -> np.ndarray:
    s = int(image_size)
    img = rng.normal(loc=0.0, scale=float(noise_std), size=(s, s)).astype(np.float32)
    img = np.clip(img, -1.0, 1.0)
    size = int(rng.integers(low=int(square_min), high=int(square_max) + 1))
    top = int(rng.integers(low=0, high=max(1, s - size)))
    left = int(rng.integers(low=0, high=max(1, s - size)))
    img[top : top + size, left : left + size] = 1.0
    return img


def _make_stripes(rng: np.random.Generator, *, image_size: int, noise_std: float, stripe_period: int) -> np.ndarray:
    s = int(image_size)
    yy, xx = np.meshgrid(np.arange(s, dtype=np.float32), np.arange(s, dtype=np.float32), indexing="ij")
    period = max(2, int(stripe_period))
    stripes = np.sin(2.0 * np.pi * (xx + yy * 0.25) / float(period))
    stripes = (stripes > 0).astype(np.float32) * 2.0 - 1.0
    noise = rng.normal(loc=0.0, scale=float(noise_std), size=(s, s)).astype(np.float32)
    img = np.clip(stripes * 0.75 + noise * 0.25, -1.0, 1.0)
    return img


class SyntheticUnpairedDomains:
    def __init__(self, cfg: DataConfig) -> None:
        if int(cfg.image_size) < 16:
            raise ValueError("image_size must be >= 16")
        if int(cfg.in_channels) <= 0:
            raise ValueError("in_channels must be > 0")
        if not (0.0 < float(cfg.val_fraction) < 1.0):
            raise ValueError("val_fraction must be in (0, 1)")
        self.cfg = cfg

    def __len__(self) -> int:
        return int(self.cfg.num_samples)

    def __getitem__(self, idx: int):
        import torch

        idx = int(idx)
        cfg = self.cfg
        rng = np.random.default_rng(int(cfg.seed) * 1_000_003 + idx)
        a = _make_square(
            rng,
            image_size=int(cfg.image_size),
            noise_std=float(cfg.noise_std),
            square_min=int(cfg.square_min),
            square_max=int(cfg.square_max),
        )
        b = _make_stripes(
            rng,
            image_size=int(cfg.image_size),
            noise_std=float(cfg.noise_std),
            stripe_period=int(cfg.stripe_period),
        )
        c = int(cfg.in_channels)
        a_t = torch.from_numpy(a).unsqueeze(0).repeat(c, 1, 1)
        b_t = torch.from_numpy(b).unsqueeze(0).repeat(c, 1, 1)
        return a_t.to(torch.float32), b_t.to(torch.float32)


def get_dataloaders(cfg: DataConfig):
    from torch.utils.data import DataLoader, Subset

    ds = SyntheticUnpairedDomains(cfg)
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


__all__ = ["DataConfig", "SyntheticUnpairedDomains", "get_dataloaders"]

