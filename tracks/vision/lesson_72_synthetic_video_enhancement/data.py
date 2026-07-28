from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from dlhub.data.splits import train_val_split_indices


@dataclass(frozen=True)
class DataConfig:
    num_samples: int = 1024
    batch_size: int = 16
    seq_len: int = 5
    image_size: int = 32
    val_fraction: float = 0.2
    seed: int = 0
    num_workers: int = 0
    in_channels: int = 1
    noise_std: float = 0.04
    blur_kernel_size: int = 3


class SyntheticVideoEnhancementDataset(Dataset):
    """Paired synthetic clips for low-quality to clean enhancement."""

    def __init__(self, cfg: DataConfig) -> None:
        self.cfg = cfg
        if int(cfg.seq_len) < 3:
            raise ValueError("seq_len must be >= 3")
        if int(cfg.image_size) < 16:
            raise ValueError("image_size must be >= 16")
        if int(cfg.in_channels) != 1:
            raise ValueError("in_channels must be 1 for this synthetic lesson")
        if float(cfg.noise_std) < 0.0:
            raise ValueError("noise_std must be >= 0")
        if int(cfg.blur_kernel_size) < 1 or int(cfg.blur_kernel_size) % 2 == 0:
            raise ValueError("blur_kernel_size must be odd and >= 1")
        if not (0.0 < float(cfg.val_fraction) < 1.0):
            raise ValueError("val_fraction must be in (0, 1)")

    def __len__(self) -> int:
        return int(self.cfg.num_samples)

    def _generator(self, idx: int) -> torch.Generator:
        seed = int(self.cfg.seed) * 1_000_003 + int(idx) * 563 + 31
        return torch.Generator().manual_seed(seed)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        cfg = self.cfg
        gen = self._generator(int(idx))
        seq_len = int(cfg.seq_len)
        size = int(cfg.image_size)

        yy, xx = torch.meshgrid(
            torch.arange(size, dtype=torch.float32),
            torch.arange(size, dtype=torch.float32),
            indexing="ij",
        )
        x_norm = xx / max(1.0, float(size - 1))
        y_norm = yy / max(1.0, float(size - 1))

        clean_clip = torch.zeros((seq_len, 1, size, size), dtype=torch.float32)
        for t in range(seq_len):
            progress = float(t) / float(max(1, seq_len - 1))
            background = 0.08 + 0.08 * x_norm + 0.10 * (1.0 - y_norm)
            center_x = size * (0.18 + 0.64 * progress)
            center_y = size * (0.32 + 0.30 * torch.sin(torch.tensor(progress * 3.14159)).item())

            square = (
                (torch.abs(xx - center_x) <= size / 9.0)
                & (torch.abs(yy - center_y) <= size / 11.0)
            ).to(torch.float32)
            halo = torch.exp(
                -0.5
                * (
                    torch.square((xx - center_x) / max(1.0, size / 8.0))
                    + torch.square((yy - center_y) / max(1.0, size / 7.0))
                )
            )
            frame = (background + 0.45 * square + 0.25 * halo).clamp(0.0, 1.0)
            clean_clip[t, 0] = frame

        kernel = int(cfg.blur_kernel_size)
        blurred = F.avg_pool2d(
            clean_clip.view(seq_len, 1, size, size),
            kernel_size=kernel,
            stride=1,
            padding=kernel // 2,
        )
        noise = torch.randn((seq_len, 1, size, size), generator=gen, dtype=torch.float32) * float(cfg.noise_std)
        degraded_clip = (0.25 * clean_clip + 0.75 * blurred + noise).clamp(0.0, 1.0)
        return degraded_clip, {"clean": clean_clip}


def get_dataloaders(cfg: DataConfig) -> tuple[DataLoader, DataLoader]:
    ds = SyntheticVideoEnhancementDataset(cfg)
    train_idx, val_idx = train_val_split_indices(
        n=len(ds),
        val_fraction=float(cfg.val_fraction),
        seed=int(cfg.seed),
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


__all__ = ["DataConfig", "SyntheticVideoEnhancementDataset", "get_dataloaders"]
