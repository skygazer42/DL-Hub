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
    seq_len: int = 6
    image_size: int = 32
    val_fraction: float = 0.2
    seed: int = 0
    num_workers: int = 0
    in_channels: int = 1
    max_jitter: int = 2
    noise_std: float = 0.02


class SyntheticVideoStabilizationDataset(Dataset):
    def __init__(self, cfg: DataConfig) -> None:
        self.cfg = cfg
        if int(cfg.seq_len) < 3:
            raise ValueError("seq_len must be >= 3")
        if int(cfg.image_size) < 16:
            raise ValueError("image_size must be >= 16")
        if int(cfg.in_channels) != 1:
            raise ValueError("in_channels must be 1 for this synthetic lesson")
        if int(cfg.max_jitter) < 0:
            raise ValueError("max_jitter must be >= 0")

    def __len__(self) -> int:
        return int(self.cfg.num_samples)

    def _generator(self, idx: int) -> torch.Generator:
        seed = int(self.cfg.seed) * 1_000_003 + int(idx) * 557 + 19
        return torch.Generator().manual_seed(seed)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        cfg = self.cfg
        gen = self._generator(idx)
        seq_len = int(cfg.seq_len)
        size = int(cfg.image_size)
        yy, xx = torch.meshgrid(
            torch.arange(size, dtype=torch.float32),
            torch.arange(size, dtype=torch.float32),
            indexing="ij",
        )

        stabilized = torch.zeros((seq_len, 1, size, size), dtype=torch.float32)
        for t in range(seq_len):
            p = float(t) / float(max(1, seq_len - 1))
            cx = size * (0.2 + 0.55 * p)
            cy = size * (0.7 - 0.35 * p)
            radius = max(2.0, size / 11.0)
            dist2 = torch.square(xx - cx) + torch.square(yy - cy)
            blob = torch.exp(-0.5 * dist2 / (radius**2))
            stabilized[t, 0] = (0.08 + 0.85 * blob).clamp(0.0, 1.0)

        jittered = torch.zeros_like(stabilized)
        max_jitter = int(cfg.max_jitter)
        for t in range(seq_len):
            shift_x = int(torch.randint(-max_jitter, max_jitter + 1, (1,), generator=gen).item())
            shift_y = int(torch.randint(-max_jitter, max_jitter + 1, (1,), generator=gen).item())
            frame = stabilized[t]
            shifted = torch.roll(frame, shifts=(shift_y, shift_x), dims=(1, 2))
            jittered[t] = shifted

        jittered = F.avg_pool2d(jittered, kernel_size=3, stride=1, padding=1)
        noise = torch.randn(jittered.shape, generator=gen, dtype=torch.float32) * float(cfg.noise_std)
        jittered = (jittered + noise).clamp(0.0, 1.0)
        return jittered, {"stabilized": stabilized}


def get_dataloaders(cfg: DataConfig) -> tuple[DataLoader, DataLoader]:
    ds = SyntheticVideoStabilizationDataset(cfg)
    train_idx, val_idx = train_val_split_indices(n=len(ds), val_fraction=float(cfg.val_fraction), seed=int(cfg.seed))
    train_loader = DataLoader(Subset(ds, train_idx), batch_size=int(cfg.batch_size), shuffle=True, num_workers=int(cfg.num_workers))
    val_loader = DataLoader(Subset(ds, val_idx), batch_size=int(cfg.batch_size), shuffle=False, num_workers=int(cfg.num_workers))
    return train_loader, val_loader


__all__ = ["DataConfig", "SyntheticVideoStabilizationDataset", "get_dataloaders"]
