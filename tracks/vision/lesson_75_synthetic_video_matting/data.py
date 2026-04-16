from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from dlhub.data.splits import train_val_split_indices


@dataclass(frozen=True)
class DataConfig:
    num_samples: int = 1024
    batch_size: int = 8
    seq_len: int = 4
    image_size: int = 32
    val_fraction: float = 0.2
    seed: int = 0
    num_workers: int = 0
    in_channels: int = 1
    noise_std: float = 0.02


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
        raise ValueError("in_channels must be 1 for this toy lesson")
    if float(cfg.noise_std) < 0.0:
        raise ValueError("noise_std must be >= 0")


def _make_frame_alpha(
    *,
    size: int,
    center_x: float,
    center_y: float,
    radius: float,
) -> torch.Tensor:
    yy = torch.arange(size, dtype=torch.float32).view(size, 1)
    xx = torch.arange(size, dtype=torch.float32).view(1, size)
    dist = torch.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2)
    alpha = (1.0 - dist / max(1.0, radius)).clamp(0.0, 1.0)
    return alpha


def make_video_matting_sample(cfg: DataConfig, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _validate_config(cfg)
    generator = torch.Generator().manual_seed(int(cfg.seed) * 1_000_003 + int(idx) * 97 + 19)
    size = int(cfg.image_size)
    seq_len = int(cfg.seq_len)

    background = torch.linspace(0.1, 0.5, steps=size, dtype=torch.float32).view(size, 1).expand(size, size)
    fg_base = torch.linspace(0.6, 1.0, steps=size, dtype=torch.float32).view(1, size).expand(size, size)

    base_radius = float(size) * float((0.16 + 0.08 * torch.rand((1,), generator=generator)).item())
    cx0 = float((0.2 * size + 0.6 * size * torch.rand((1,), generator=generator)).item())
    cy0 = float((0.2 * size + 0.6 * size * torch.rand((1,), generator=generator)).item())
    vx = float((-1.4 + 2.8 * torch.rand((1,), generator=generator)).item())
    vy = float((-1.4 + 2.8 * torch.rand((1,), generator=generator)).item())

    video = []
    trimap = []
    alpha = []
    for t in range(seq_len):
        center_x = float(torch.clamp(torch.tensor(cx0 + vx * t), 2.0, float(size - 3)).item())
        center_y = float(torch.clamp(torch.tensor(cy0 + vy * t), 2.0, float(size - 3)).item())
        alpha_frame = _make_frame_alpha(
            size=size,
            center_x=center_x,
            center_y=center_y,
            radius=base_radius,
        )
        fg = (fg_base + 0.15 * torch.sin((torch.arange(size, dtype=torch.float32).view(1, size) + t) / 5.0)).clamp(
            0.0, 1.0
        )
        frame = alpha_frame * fg + (1.0 - alpha_frame) * background
        if float(cfg.noise_std) > 0.0:
            frame = frame + torch.randn((size, size), generator=generator, dtype=torch.float32) * float(cfg.noise_std)
        frame = frame.clamp(0.0, 1.0)

        tri = torch.full((size, size), 0.5, dtype=torch.float32)
        tri = torch.where(alpha_frame >= 0.9, torch.ones_like(tri), tri)
        tri = torch.where(alpha_frame <= 0.1, torch.zeros_like(tri), tri)

        video.append(frame.unsqueeze(0))
        trimap.append(tri.unsqueeze(0))
        alpha.append(alpha_frame.unsqueeze(0))

    return torch.stack(video, dim=0), torch.stack(trimap, dim=0), torch.stack(alpha, dim=0)


class SyntheticVideoMattingDataset(Dataset):
    def __init__(self, cfg: DataConfig) -> None:
        _validate_config(cfg)
        self.cfg = cfg

    def __len__(self) -> int:
        return int(self.cfg.num_samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return make_video_matting_sample(self.cfg, int(idx))


def get_dataloaders(cfg: DataConfig) -> tuple[DataLoader, DataLoader]:
    _validate_config(cfg)
    dataset = SyntheticVideoMattingDataset(cfg)
    train_idx, val_idx = train_val_split_indices(
        n=len(dataset),
        val_fraction=float(cfg.val_fraction),
        seed=int(cfg.seed),
    )
    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=int(cfg.batch_size),
        shuffle=True,
        num_workers=int(cfg.num_workers),
        pin_memory=False,
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=int(cfg.batch_size),
        shuffle=False,
        num_workers=int(cfg.num_workers),
        pin_memory=False,
    )
    return train_loader, val_loader


__all__ = [
    "DataConfig",
    "SyntheticVideoMattingDataset",
    "get_dataloaders",
    "make_video_matting_sample",
]
