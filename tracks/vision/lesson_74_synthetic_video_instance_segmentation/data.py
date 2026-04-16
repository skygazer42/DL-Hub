from __future__ import annotations

from dataclasses import dataclass

import torch
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
    num_instances: int = 2
    noise_std: float = 0.02


class SyntheticVideoInstanceSegmentationDataset(Dataset):
    """Render fixed-slot toy instances with per-frame masks."""

    def __init__(self, cfg: DataConfig) -> None:
        self.cfg = cfg
        if int(cfg.seq_len) < 3:
            raise ValueError("seq_len must be >= 3")
        if int(cfg.image_size) < 16:
            raise ValueError("image_size must be >= 16")
        if int(cfg.in_channels) != 1:
            raise ValueError("in_channels must be 1 for this toy lesson")
        if int(cfg.num_instances) < 1:
            raise ValueError("num_instances must be >= 1")
        if float(cfg.noise_std) < 0.0:
            raise ValueError("noise_std must be >= 0")
        if not (0.0 < float(cfg.val_fraction) < 1.0):
            raise ValueError("val_fraction must be in (0, 1)")

    def __len__(self) -> int:
        return int(self.cfg.num_samples)

    def _generator(self, idx: int) -> torch.Generator:
        seed = int(self.cfg.seed) * 1_000_003 + int(idx) * 677 + 41
        return torch.Generator().manual_seed(seed)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        cfg = self.cfg
        gen = self._generator(int(idx))
        seq_len = int(cfg.seq_len)
        size = int(cfg.image_size)
        num_instances = int(cfg.num_instances)

        yy, xx = torch.meshgrid(
            torch.arange(size, dtype=torch.float32),
            torch.arange(size, dtype=torch.float32),
            indexing="ij",
        )
        x_norm = xx / max(1.0, float(size - 1))
        y_norm = yy / max(1.0, float(size - 1))

        clip = torch.zeros((seq_len, 1, size, size), dtype=torch.float32)
        masks = torch.zeros((seq_len, num_instances, size, size), dtype=torch.float32)

        for slot in range(num_instances):
            base_y = size * (0.20 + 0.60 * slot / max(1, num_instances - 1 if num_instances > 1 else 1))
            radius_x = max(2.0, size / 10.0)
            radius_y = max(2.0, size / 12.0)
            direction = 1.0 if slot % 2 == 0 else -1.0
            for t in range(seq_len):
                progress = float(t) / float(max(1, seq_len - 1))
                center_x = size * (0.18 + 0.64 * progress)
                if direction < 0:
                    center_x = size * (0.82 - 0.64 * progress)
                center_y = base_y + 2.0 * torch.sin(torch.tensor(progress * 3.14159 + slot)).item()
                mask = (
                    torch.square((xx - center_x) / radius_x)
                    + torch.square((yy - center_y) / radius_y)
                    <= 1.0
                ).to(torch.float32)
                masks[t, slot] = mask
                clip[t, 0] = torch.maximum(clip[t, 0], mask * (0.40 + 0.15 * slot))

        noise = torch.randn((seq_len, size, size), generator=gen, dtype=torch.float32) * float(cfg.noise_std)
        background = (0.05 + 0.08 * x_norm + 0.06 * (1.0 - y_norm)).unsqueeze(0)
        clip[:, 0] = (clip[:, 0] + background + noise).clamp(0.0, 1.0)
        return clip, {"instance_masks": masks}


def get_dataloaders(cfg: DataConfig) -> tuple[DataLoader, DataLoader]:
    ds = SyntheticVideoInstanceSegmentationDataset(cfg)
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


__all__ = ["DataConfig", "SyntheticVideoInstanceSegmentationDataset", "get_dataloaders"]
