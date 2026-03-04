from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from dlhub.data.splits import train_val_split_indices


@dataclass(frozen=True)
class DataConfig:
    num_samples: int = 2048
    batch_size: int = 32
    image_size: int = 64
    val_fraction: float = 0.2
    seed: int = 0
    num_workers: int = 0

    in_channels: int = 1
    noise_std: float = 0.1
    min_square: int = 8
    max_square: int = 24
    train_mode: str = "supervised"  # supervised | noise2noise


class ToyDenoisingSquares(Dataset):
    """Toy denoising dataset (clean squares + additive Gaussian noise)."""

    def __init__(self, cfg: DataConfig, *, mode: str) -> None:
        self.cfg = cfg
        self.mode = str(mode).lower().strip()
        if self.mode not in {"supervised", "noise2noise"}:
            raise ValueError(f"Unknown mode: {mode!r}. Expected 'supervised' | 'noise2noise'.")

        s = int(cfg.image_size)
        if s < 16:
            raise ValueError("image_size must be >= 16")
        if int(cfg.min_square) < 2 or int(cfg.max_square) < int(cfg.min_square):
            raise ValueError("invalid square size range")
        if not (0.0 < float(cfg.val_fraction) < 1.0):
            raise ValueError("val_fraction must be in (0, 1)")

        n = int(cfg.num_samples)
        g = torch.Generator().manual_seed(int(cfg.seed))

        sizes = torch.randint(
            low=int(cfg.min_square),
            high=int(cfg.max_square) + 1,
            size=(n,),
            generator=g,
            dtype=torch.long,
        )

        tops = torch.empty((n,), dtype=torch.long)
        lefts = torch.empty((n,), dtype=torch.long)
        for i in range(n):
            size = int(sizes[i].item())
            top = int(torch.randint(low=0, high=max(1, s - size + 1), size=(1,), generator=g).item())
            left = int(torch.randint(low=0, high=max(1, s - size + 1), size=(1,), generator=g).item())
            tops[i] = top
            lefts[i] = left

        self.sizes = sizes
        self.tops = tops
        self.lefts = lefts

    def __len__(self) -> int:
        return int(self.sizes.numel())

    def _clean(self, idx: int) -> torch.Tensor:
        cfg = self.cfg
        s = int(cfg.image_size)
        c = int(cfg.in_channels)
        size = int(self.sizes[idx].item())
        top = int(self.tops[idx].item())
        left = int(self.lefts[idx].item())

        img = torch.zeros((1, s, s), dtype=torch.float32)
        img[:, top : top + size, left : left + size] = 1.0
        if c == 1:
            return img
        return img.repeat(c, 1, 1)

    def _noise(self, idx: int, *, stream: int) -> torch.Tensor:
        # Per-sample RNG: deterministic across dataloader workers.
        g = torch.Generator().manual_seed(int(self.cfg.seed) * 1_000_003 + int(idx) * 17 + int(stream))
        c = int(self.cfg.in_channels)
        s = int(self.cfg.image_size)
        return torch.randn((c, s, s), generator=g, dtype=torch.float32) * float(self.cfg.noise_std)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        i = int(idx)
        clean = self._clean(i)

        if self.mode == "supervised":
            noisy = (clean + self._noise(i, stream=0)).clamp(0.0, 1.0)
            return noisy, clean

        # noise2noise: two independent noise realizations of the same clean signal
        noisy1 = (clean + self._noise(i, stream=0)).clamp(0.0, 1.0)
        noisy2 = (clean + self._noise(i, stream=1)).clamp(0.0, 1.0)
        return noisy1, noisy2


def get_dataloaders(cfg: DataConfig) -> tuple[DataLoader, DataLoader]:
    """Return `(train_loader, val_loader)` for the toy denoising task."""

    train_mode = str(cfg.train_mode).lower().strip()
    train_ds_full = ToyDenoisingSquares(cfg, mode=train_mode)
    val_ds_full = ToyDenoisingSquares(cfg, mode="supervised")

    train_idx, val_idx = train_val_split_indices(n=len(train_ds_full), val_fraction=cfg.val_fraction, seed=cfg.seed)
    train_ds = Subset(train_ds_full, train_idx)
    val_ds = Subset(val_ds_full, val_idx)

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg.batch_size),
        shuffle=True,
        num_workers=int(cfg.num_workers),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg.batch_size),
        shuffle=False,
        num_workers=int(cfg.num_workers),
        drop_last=False,
    )
    return train_loader, val_loader


__all__ = ["DataConfig", "ToyDenoisingSquares", "get_dataloaders"]

