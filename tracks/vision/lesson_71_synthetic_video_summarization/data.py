from __future__ import annotations

from dataclasses import dataclass

import torch
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
    num_key_frames: int = 2
    noise_std: float = 0.02


class SyntheticVideoSummarizationDataset(Dataset):
    """Score which frames best summarize a compact synthetic clip."""

    def __init__(self, cfg: DataConfig) -> None:
        self.cfg = cfg
        if int(cfg.seq_len) < 4:
            raise ValueError("seq_len must be >= 4")
        if int(cfg.image_size) < 16:
            raise ValueError("image_size must be >= 16")
        if int(cfg.in_channels) != 1:
            raise ValueError("in_channels must be 1 for this synthetic lesson")
        if not (1 <= int(cfg.num_key_frames) < int(cfg.seq_len)):
            raise ValueError("num_key_frames must be in [1, seq_len)")
        if float(cfg.noise_std) < 0.0:
            raise ValueError("noise_std must be >= 0")
        if not (0.0 < float(cfg.val_fraction) < 1.0):
            raise ValueError("val_fraction must be in (0, 1)")

    def __len__(self) -> int:
        return int(self.cfg.num_samples)

    def _generator(self, idx: int) -> torch.Generator:
        seed = int(self.cfg.seed) * 1_000_003 + int(idx) * 421 + 23
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

        key_frames = torch.randperm(seq_len, generator=gen)[: int(cfg.num_key_frames)]
        key_frames = torch.sort(key_frames).values
        importance = torch.zeros((seq_len,), dtype=torch.float32)
        importance[key_frames] = 1.0

        clip = torch.zeros((seq_len, 1, size, size), dtype=torch.float32)
        base_radius = max(1.5, size / 18.0)
        for t in range(seq_len):
            progress = float(t) / float(max(1, seq_len - 1))
            cx = size * (0.15 + 0.7 * progress)
            cy = size * 0.35
            background = 0.05 + 0.10 * x_norm + 0.08 * (1.0 - y_norm)

            dist2 = torch.square(xx - cx) + torch.square(yy - cy) * 1.2
            moving_blob = torch.exp(-0.5 * dist2 / (base_radius**2))

            if importance[t].item() > 0.5:
                key_square = (
                    (torch.abs(xx - size * 0.65) <= size / 10.0)
                    & (torch.abs(yy - size * 0.68) <= size / 10.0)
                ).to(torch.float32)
                moving_blob = moving_blob + 0.9 * key_square

            noise = torch.randn((size, size), generator=gen, dtype=torch.float32) * float(cfg.noise_std)
            frame = (background + 0.42 * moving_blob + noise).clamp(0.0, 1.0)
            clip[t, 0] = frame

        return clip, {"importance": importance}


def get_dataloaders(cfg: DataConfig) -> tuple[DataLoader, DataLoader]:
    ds = SyntheticVideoSummarizationDataset(cfg)
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


__all__ = ["DataConfig", "SyntheticVideoSummarizationDataset", "get_dataloaders"]
