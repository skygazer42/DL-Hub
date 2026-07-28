from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from dlhub.data.splits import train_val_split_indices


@dataclass(frozen=True)
class DataConfig:
    batch_size: int = 8
    image_size: int = 16
    num_frames: int = 4
    num_workers: int = 0
    num_samples: int = 64
    seed: int = 0
    val_fraction: float = 0.2


def _make_source_frame(image_size: int, generator: torch.Generator) -> torch.Tensor:
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, image_size, dtype=torch.float32),
        torch.linspace(-1.0, 1.0, image_size, dtype=torch.float32),
        indexing="ij",
    )
    cx = float(torch.empty((1,)).uniform_(-0.45, 0.45, generator=generator).item())
    cy = float(torch.empty((1,)).uniform_(-0.45, 0.45, generator=generator).item())
    radius = float(torch.empty((1,)).uniform_(0.22, 0.48, generator=generator).item())
    blob = torch.exp(-((xx - cx).pow(2) + (yy - cy).pow(2)) / (2.0 * radius * radius))

    stripe = 0.5 + 0.5 * torch.sin(4.0 * xx - 2.0 * yy)
    ring = 0.5 + 0.5 * torch.cos((xx.pow(2) + yy.pow(2)).sqrt() * 8.0)
    frame = torch.stack([blob, stripe, ring], dim=0)
    frame += 0.03 * torch.rand((3, image_size, image_size), generator=generator, dtype=torch.float32)
    return frame.clamp(0.0, 1.0)


def _make_video_pair(
    image_size: int,
    num_frames: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    base = _make_source_frame(image_size, generator)
    dy = int(torch.randint(-2, 3, (1,), generator=generator).item())
    dx = int(torch.randint(-2, 3, (1,), generator=generator).item())
    color_gain = torch.empty((3, 1, 1)).uniform_(0.85, 1.15, generator=generator)
    color_bias = torch.empty((3, 1, 1)).uniform_(-0.08, 0.08, generator=generator)

    source_frames: list[torch.Tensor] = []
    target_frames: list[torch.Tensor] = []
    positions = torch.linspace(-1.0, 1.0, int(num_frames), dtype=torch.float32)
    for frame_idx, alpha in enumerate(positions):
        sy = int(round(float(alpha.item()) * dy))
        sx = int(round(float(alpha.item()) * dx))
        source = torch.roll(base, shifts=(sy, sx), dims=(1, 2))

        target = source * color_gain + color_bias
        target += 0.05 * torch.sin(torch.tensor(frame_idx * 0.9, dtype=torch.float32))
        target = torch.roll(target, shifts=(1, -1), dims=(1, 2))
        target += 0.02 * torch.randn((3, image_size, image_size), generator=generator, dtype=torch.float32)

        source_frames.append(source.clamp(0.0, 1.0))
        target_frames.append(target.clamp(0.0, 1.0))

    source_video = torch.stack(source_frames, dim=1)
    target_video = torch.stack(target_frames, dim=1)
    return source_video, target_video


class SyntheticVideoToVideoDataset(Dataset):
    def __init__(self, cfg: DataConfig) -> None:
        self.cfg = cfg
        generator = torch.Generator().manual_seed(int(cfg.seed))
        source_videos: list[torch.Tensor] = []
        target_videos: list[torch.Tensor] = []
        for _ in range(int(cfg.num_samples)):
            source, target = _make_video_pair(
                image_size=int(cfg.image_size),
                num_frames=int(cfg.num_frames),
                generator=generator,
            )
            source_videos.append(source)
            target_videos.append(target)
        self._source = torch.stack(source_videos, dim=0)
        self._target = torch.stack(target_videos, dim=0)

    def __len__(self) -> int:
        return int(self._source.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        i = int(idx)
        return self._source[i], self._target[i]


def get_dataloaders(cfg: DataConfig) -> tuple[DataLoader, DataLoader]:
    dataset: Dataset = SyntheticVideoToVideoDataset(cfg)
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
        drop_last=False,
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=int(cfg.batch_size),
        shuffle=False,
        num_workers=int(cfg.num_workers),
        drop_last=False,
    )
    return train_loader, val_loader


__all__ = ["DataConfig", "SyntheticVideoToVideoDataset", "get_dataloaders"]
