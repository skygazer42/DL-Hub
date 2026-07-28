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
    num_classes: int = 3
    max_objects: int = 2
    noise_std: float = 0.02


class SyntheticVideoObjectDetectionDataset(Dataset):
    def __init__(self, cfg: DataConfig) -> None:
        self.cfg = cfg
        if int(cfg.seq_len) < 3:
            raise ValueError("seq_len must be >= 3")
        if int(cfg.image_size) < 16:
            raise ValueError("image_size must be >= 16")
        if int(cfg.in_channels) != 1:
            raise ValueError("in_channels must be 1 for this synthetic lesson")
        if int(cfg.max_objects) < 1:
            raise ValueError("max_objects must be >= 1")
        if int(cfg.num_classes) < 2:
            raise ValueError("num_classes must be >= 2")

    def __len__(self) -> int:
        return int(self.cfg.num_samples)

    def _generator(self, idx: int) -> torch.Generator:
        seed = int(self.cfg.seed) * 1_000_003 + int(idx) * 313 + 13
        return torch.Generator().manual_seed(seed)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        cfg = self.cfg
        gen = self._generator(idx)
        seq_len = int(cfg.seq_len)
        size = int(cfg.image_size)
        max_objects = int(cfg.max_objects)

        yy, xx = torch.meshgrid(
            torch.arange(size, dtype=torch.float32),
            torch.arange(size, dtype=torch.float32),
            indexing="ij",
        )
        clip = torch.zeros((seq_len, 1, size, size), dtype=torch.float32)
        boxes = torch.zeros((max_objects, 4), dtype=torch.float32)
        labels = torch.zeros((max_objects,), dtype=torch.int64)
        present = torch.zeros((max_objects,), dtype=torch.float32)

        num_active = int(torch.randint(1, max_objects + 1, (1,), generator=gen).item())
        for slot in range(num_active):
            present[slot] = 1.0
            labels[slot] = int(torch.randint(0, int(cfg.num_classes), (1,), generator=gen).item())
            bw = float(torch.randint(size // 7, size // 4, (1,), generator=gen).item())
            bh = float(torch.randint(size // 7, size // 4, (1,), generator=gen).item())
            x1 = float(torch.randint(1, max(2, size - int(bw) - 1), (1,), generator=gen).item())
            y1 = float(torch.randint(1, max(2, size - int(bh) - 1), (1,), generator=gen).item())
            vx = (slot + 1) * 0.6 * (1.0 if slot % 2 == 0 else -1.0)
            vy = (slot + 1) * 0.35

            for t in range(seq_len):
                cx = x1 + bw / 2.0 + vx * t
                cy = y1 + bh / 2.0 + vy * t
                cur_x1 = max(0.0, min(float(size - 2), cx - bw / 2.0))
                cur_y1 = max(0.0, min(float(size - 2), cy - bh / 2.0))
                cur_x2 = max(cur_x1 + 1.0, min(float(size - 1), cur_x1 + bw))
                cur_y2 = max(cur_y1 + 1.0, min(float(size - 1), cur_y1 + bh))
                mask = (xx >= cur_x1) & (xx <= cur_x2) & (yy >= cur_y1) & (yy <= cur_y2)
                clip[t, 0] = torch.where(mask, torch.full_like(clip[t, 0], 0.35 + 0.2 * slot), clip[t, 0])

            boxes[slot] = torch.tensor([x1 / size, y1 / size, (x1 + bw) / size, (y1 + bh) / size], dtype=torch.float32)

        noise = torch.randn(clip.shape, generator=gen, dtype=torch.float32)
        clip = (clip + noise * float(cfg.noise_std)).clamp(0.0, 1.0)
        return clip, {"boxes": boxes, "labels": labels, "present": present}


def get_dataloaders(cfg: DataConfig) -> tuple[DataLoader, DataLoader]:
    ds = SyntheticVideoObjectDetectionDataset(cfg)
    train_idx, val_idx = train_val_split_indices(
        n=len(ds),
        val_fraction=float(cfg.val_fraction),
        seed=int(cfg.seed),
    )
    train_loader = DataLoader(Subset(ds, train_idx), batch_size=int(cfg.batch_size), shuffle=True, num_workers=int(cfg.num_workers))
    val_loader = DataLoader(Subset(ds, val_idx), batch_size=int(cfg.batch_size), shuffle=False, num_workers=int(cfg.num_workers))
    return train_loader, val_loader


__all__ = ["DataConfig", "SyntheticVideoObjectDetectionDataset", "get_dataloaders"]
