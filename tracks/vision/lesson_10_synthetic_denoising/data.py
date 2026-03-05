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
    train_mode: str = "supervised"  # supervised | noise2noise | blindspot
    blindspot_prob: float = 0.1  # fraction of pixels to mask for blind-spot training


class ToyDenoisingSquares(Dataset):
    """Toy denoising dataset (clean squares + additive Gaussian noise)."""

    def __init__(self, cfg: DataConfig, *, mode: str) -> None:
        self.cfg = cfg
        self.mode = str(mode).lower().strip()
        if self.mode not in {"supervised", "noise2noise", "blindspot"}:
            raise ValueError(f"Unknown mode: {mode!r}. Expected 'supervised' | 'noise2noise' | 'blindspot'.")

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

    def _blindspot_mask(self, idx: int) -> torch.Tensor:
        cfg = self.cfg
        s = int(cfg.image_size)
        c = int(cfg.in_channels)
        p = float(cfg.blindspot_prob)
        if not (0.0 < p < 1.0):
            raise ValueError("blindspot_prob must be in (0, 1)")

        g = torch.Generator().manual_seed(int(cfg.seed) * 1_000_003 + int(idx) * 17 + 2)
        m = (torch.rand((1, s, s), generator=g, dtype=torch.float32) < p).to(torch.float32)
        if c == 1:
            return m
        return m.repeat(c, 1, 1)

    def _blindspot_masked_input(self, noisy: torch.Tensor, idx: int) -> torch.Tensor:
        """Noise2Void-style masking: replace masked pixels with random neighbor pixels."""

        if noisy.ndim != 3:
            raise ValueError(f"Expected (C, H, W) noisy tensor, got {tuple(noisy.shape)}")

        mask = self._blindspot_mask(idx)  # (C, H, W)
        if mask.shape != noisy.shape:
            raise ValueError("blindspot mask shape mismatch")

        # Randomly choose replacement direction per pixel (shared across channels).
        cfg = self.cfg
        s = int(cfg.image_size)
        g = torch.Generator().manual_seed(int(cfg.seed) * 1_000_003 + int(idx) * 17 + 3)
        sel = torch.randint(0, 4, (1, s, s), generator=g, dtype=torch.long)

        up = torch.roll(noisy, shifts=1, dims=1)
        down = torch.roll(noisy, shifts=-1, dims=1)
        left = torch.roll(noisy, shifts=1, dims=2)
        right = torch.roll(noisy, shifts=-1, dims=2)

        rep = torch.where(sel == 0, up, torch.where(sel == 1, down, torch.where(sel == 2, left, right)))
        masked = noisy * (1.0 - mask) + rep * mask
        return masked

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        i = int(idx)
        clean = self._clean(i)

        if self.mode == "supervised":
            noisy = (clean + self._noise(i, stream=0)).clamp(0.0, 1.0)
            return noisy, clean

        # noise2noise: two independent noise realizations of the same clean signal
        noisy1 = (clean + self._noise(i, stream=0)).clamp(0.0, 1.0)
        noisy2 = (clean + self._noise(i, stream=1)).clamp(0.0, 1.0)
        if self.mode == "noise2noise":
            return noisy1, noisy2

        # blindspot: self-supervised via masking (target is the *noisy* image; loss only on masked pixels)
        noisy = noisy1
        masked_noisy = self._blindspot_masked_input(noisy, i)
        mask = self._blindspot_mask(i)
        return masked_noisy, {"target": noisy, "mask": mask}


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
