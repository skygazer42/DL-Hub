from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from dlhub.data.splits import train_val_split_indices


@dataclass(frozen=True)
class DataConfig:
    num_samples: int = 1024
    batch_size: int = 16
    seq_len: int = 8
    image_size: int = 32
    val_fraction: float = 0.2
    seed: int = 0
    num_workers: int = 0
    in_channels: int = 1
    num_classes: int = 4
    noise_std: float = 0.02
    motion_jitter: float = 0.12


class SyntheticActionRecognitionDataset(Dataset):
    """Classify compact clips by deterministic motion trajectories."""

    def __init__(self, cfg: DataConfig) -> None:
        self.cfg = cfg
        if int(cfg.seq_len) < 4:
            raise ValueError("seq_len must be >= 4")
        if int(cfg.image_size) < 16:
            raise ValueError("image_size must be >= 16")
        if int(cfg.in_channels) != 1:
            raise ValueError("in_channels must be 1 for this toy lesson")
        if int(cfg.num_classes) != 4:
            raise ValueError("num_classes must be 4 for this toy lesson")
        if float(cfg.noise_std) < 0.0:
            raise ValueError("noise_std must be >= 0")
        if float(cfg.motion_jitter) < 0.0:
            raise ValueError("motion_jitter must be >= 0")
        if not (0.0 < float(cfg.val_fraction) < 1.0):
            raise ValueError("val_fraction must be in (0, 1)")

    def __len__(self) -> int:
        return int(self.cfg.num_samples)

    def _generator(self, idx: int) -> torch.Generator:
        seed = int(self.cfg.seed) * 1_000_003 + int(idx) * 997 + 89
        return torch.Generator().manual_seed(seed)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        cfg = self.cfg
        gen = self._generator(int(idx))
        seq_len = int(cfg.seq_len)
        size = int(cfg.image_size)
        label = int(torch.randint(0, int(cfg.num_classes), (1,), generator=gen).item())

        yy, xx = torch.meshgrid(
            torch.arange(size, dtype=torch.float32),
            torch.arange(size, dtype=torch.float32),
            indexing="ij",
        )
        x_norm = xx / max(1.0, float(size - 1))
        y_norm = yy / max(1.0, float(size - 1))

        clip = torch.zeros((seq_len, 1, size, size), dtype=torch.float32)
        radius = max(1.8, size / 13.0)

        for t in range(seq_len):
            progress = float(t) / float(max(1, seq_len - 1))
            jitter_x = float(torch.randn((1,), generator=gen, dtype=torch.float32).item()) * float(cfg.motion_jitter)
            jitter_y = float(torch.randn((1,), generator=gen, dtype=torch.float32).item()) * float(cfg.motion_jitter)

            if label == 0:
                cx = size * (0.2 + 0.6 * progress) + jitter_x
                cy = size * (0.25 + 0.1 * torch.sin(torch.tensor(progress * 3.14159)).item()) + jitter_y
            elif label == 1:
                cx = size * (0.2 + 0.6 * progress) + jitter_x
                cy = size * (0.75 - 0.1 * torch.sin(torch.tensor(progress * 3.14159)).item()) + jitter_y
            elif label == 2:
                cx = size * 0.5 + size * 0.18 * torch.cos(torch.tensor(progress * 6.28318)).item() + jitter_x
                cy = size * 0.5 + size * 0.18 * torch.sin(torch.tensor(progress * 6.28318)).item() + jitter_y
            else:
                cx = size * (0.5 + 0.2 * torch.sin(torch.tensor(progress * 12.56636)).item()) + jitter_x
                cy = size * 0.5 + jitter_y

            background = 0.05 + 0.07 * x_norm + 0.04 * (1.0 - y_norm)
            dist2 = torch.square(xx - cx) + torch.square(yy - cy)
            actor = torch.exp(-0.5 * dist2 / (radius**2))
            noise = torch.randn((size, size), generator=gen, dtype=torch.float32) * float(cfg.noise_std)
            frame = (background + 0.9 * actor + noise).clamp(0.0, 1.0)
            clip[t, 0] = frame

        target = {"action_label": torch.tensor(label, dtype=torch.int64)}
        return clip, target


def get_dataloaders(cfg: DataConfig) -> tuple[DataLoader, DataLoader]:
    ds = SyntheticActionRecognitionDataset(cfg)
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


__all__ = ["DataConfig", "SyntheticActionRecognitionDataset", "get_dataloaders"]

