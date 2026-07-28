from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from dlhub.data.splits import train_val_split_indices


@dataclass(frozen=True)
class DataConfig:
    num_samples: int = 1024
    batch_size: int = 8
    seq_len: int = 5
    image_size: int = 32
    val_fraction: float = 0.2
    seed: int = 0
    num_workers: int = 0
    in_channels: int = 1
    noise_std: float = 0.04
    min_size: int = 4
    max_size: int = 10
    max_speed: int = 2


def _validate_config(cfg: DataConfig) -> None:
    if int(cfg.num_samples) <= 0:
        raise ValueError("num_samples must be > 0")
    if int(cfg.batch_size) <= 0:
        raise ValueError("batch_size must be > 0")
    if int(cfg.seq_len) < 2:
        raise ValueError("seq_len must be >= 2")
    if int(cfg.image_size) < 16:
        raise ValueError("image_size must be >= 16")
    if not (0.0 < float(cfg.val_fraction) < 1.0):
        raise ValueError("val_fraction must be in (0, 1)")
    if int(cfg.in_channels) != 1:
        raise ValueError("in_channels must be 1 for this synthetic lesson")
    if float(cfg.noise_std) < 0.0:
        raise ValueError("noise_std must be >= 0")
    if int(cfg.min_size) < 2 or int(cfg.max_size) < int(cfg.min_size):
        raise ValueError("invalid object size range")
    if int(cfg.max_speed) < 1:
        raise ValueError("max_speed must be >= 1")


def make_video_object_segmentation_sample(
    cfg: DataConfig, idx: int
) -> tuple[torch.Tensor, torch.Tensor]:
    _validate_config(cfg)
    seq_len = int(cfg.seq_len)
    size = int(cfg.image_size)
    generator = torch.Generator().manual_seed(int(cfg.seed) * 1_000_003 + int(idx))

    video = torch.randn((seq_len, 1, size, size), generator=generator, dtype=torch.float32)
    video = video * float(cfg.noise_std) + 0.10
    masks = torch.zeros((seq_len, 1, size, size), dtype=torch.float32)

    box_h = int(
        torch.randint(int(cfg.min_size), int(cfg.max_size) + 1, (1,), generator=generator).item()
    )
    box_w = int(
        torch.randint(int(cfg.min_size), int(cfg.max_size) + 1, (1,), generator=generator).item()
    )

    margin_x = max(1, box_w // 2 + 1)
    margin_y = max(1, box_h // 2 + 1)
    cx0 = int(torch.randint(margin_x, size - margin_x, (1,), generator=generator).item())
    cy0 = int(torch.randint(margin_y, size - margin_y, (1,), generator=generator).item())

    vx = int(torch.randint(-int(cfg.max_speed), int(cfg.max_speed) + 1, (1,), generator=generator).item())
    vy = int(torch.randint(-int(cfg.max_speed), int(cfg.max_speed) + 1, (1,), generator=generator).item())
    if vx == 0 and vy == 0:
        vx = 1

    for t in range(seq_len):
        cx = max(margin_x, min(size - margin_x - 1, cx0 + t * vx))
        cy = max(margin_y, min(size - margin_y - 1, cy0 + t * vy))
        x1 = max(0, cx - box_w // 2)
        y1 = max(0, cy - box_h // 2)
        x2 = min(size, x1 + box_w)
        y2 = min(size, y1 + box_h)
        masks[t, 0, y1:y2, x1:x2] = 1.0
        video[t, 0, y1:y2, x1:x2] = torch.maximum(video[t, 0, y1:y2, x1:x2], torch.tensor(0.90))

    video = video.clamp(0.0, 1.0)
    return video, masks


class SyntheticVideoObjectSegmentationDataset(Dataset):
    def __init__(self, cfg: DataConfig) -> None:
        _validate_config(cfg)
        self.cfg = cfg

    def __len__(self) -> int:
        return int(self.cfg.num_samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return make_video_object_segmentation_sample(self.cfg, int(idx))


def get_dataloaders(cfg: DataConfig) -> tuple[DataLoader, DataLoader]:
    _validate_config(cfg)
    ds = SyntheticVideoObjectSegmentationDataset(cfg)
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
        pin_memory=False,
    )
    val_loader = DataLoader(
        Subset(ds, val_idx),
        batch_size=int(cfg.batch_size),
        shuffle=False,
        num_workers=int(cfg.num_workers),
        pin_memory=False,
    )
    return train_loader, val_loader


__all__ = [
    "DataConfig",
    "SyntheticVideoObjectSegmentationDataset",
    "get_dataloaders",
    "make_video_object_segmentation_sample",
]
