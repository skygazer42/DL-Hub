from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from dlhub.data.splits import train_val_split_indices


@dataclass(frozen=True)
class DataConfig:
    batch_size: int = 32
    image_size: int = 28
    num_workers: int = 0
    num_samples: int = 256
    seed: int = 0
    val_fraction: float = 0.2
    num_frames: int = 4


def _shape_mask(image_size: int, generator: torch.Generator) -> torch.Tensor:
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, image_size, dtype=torch.float32),
        torch.linspace(-1.0, 1.0, image_size, dtype=torch.float32),
        indexing="ij",
    )
    mask = torch.zeros((1, image_size, image_size), dtype=torch.float32)
    mode = int(torch.randint(low=0, high=4, size=(1,), generator=generator).item())

    if mode == 0:
        cx = float(torch.empty((1,)).uniform_(-0.45, 0.45, generator=generator).item())
        cy = float(torch.empty((1,)).uniform_(-0.45, 0.45, generator=generator).item())
        radius = float(torch.empty((1,)).uniform_(0.2, 0.5, generator=generator).item())
        mask[0, (xx - cx).pow(2) + (yy - cy).pow(2) <= radius**2] = 1.0
    elif mode == 1:
        y1 = int(torch.randint(2, image_size // 2, (1,), generator=generator).item())
        y2 = int(torch.randint(image_size // 2, image_size - 2, (1,), generator=generator).item())
        x1 = int(torch.randint(2, image_size // 2, (1,), generator=generator).item())
        x2 = int(torch.randint(image_size // 2, image_size - 2, (1,), generator=generator).item())
        mask[:, y1:y2, x1:x2] = 1.0
    elif mode == 2:
        center = int(torch.randint(5, image_size - 5, (1,), generator=generator).item())
        thickness = int(torch.randint(1, 3, (1,), generator=generator).item())
        mask[:, center - thickness : center + thickness + 1, :] = 1.0
        mask[:, :, center - thickness : center + thickness + 1] = 1.0
    else:
        slope = float(torch.empty((1,)).uniform_(-0.85, 0.85, generator=generator).item())
        bias = float(torch.empty((1,)).uniform_(-0.3, 0.3, generator=generator).item())
        width = float(torch.empty((1,)).uniform_(0.09, 0.24, generator=generator).item())
        mask[0, torch.abs(yy - slope * xx - bias) <= width] = 1.0

    return mask


def _texture(image_size: int, generator: torch.Generator) -> torch.Tensor:
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, image_size, dtype=torch.float32),
        torch.linspace(-1.0, 1.0, image_size, dtype=torch.float32),
        indexing="ij",
    )
    mode = int(torch.randint(0, 3, (1,), generator=generator).item())
    if mode == 0:
        tex = 0.5 + 0.5 * torch.sin(xx * 8.0 + yy * 3.0)
    elif mode == 1:
        tex = (((xx * 8.0).floor() + (yy * 8.0).floor()) % 2.0 == 0.0).to(torch.float32)
    else:
        tex = 0.5 + 0.5 * torch.cos((xx.pow(2) + yy.pow(2)).sqrt() * 10.0)
    tex = tex.unsqueeze(0)
    tex += 0.03 * torch.rand((1, image_size, image_size), generator=generator, dtype=torch.float32)
    return tex.clamp(0.0, 1.0)


def _make_source(mask: torch.Tensor, texture: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    background = 0.05 + 0.1 * torch.rand(mask.shape, generator=generator, dtype=torch.float32)
    foreground = 0.2 + 0.72 * texture
    source = background * (1.0 - mask) + foreground * mask
    source += 0.02 * torch.rand(mask.shape, generator=generator, dtype=torch.float32)
    return source.clamp(0.0, 1.0)


def _sample_motion(generator: torch.Generator) -> tuple[int, int, float]:
    choices = torch.tensor(
        [
            [-2, -1],
            [-1, 2],
            [1, -2],
            [2, 1],
            [-2, 2],
            [2, -2],
        ],
        dtype=torch.int64,
    )
    index = int(torch.randint(0, choices.shape[0], (1,), generator=generator).item())
    dy = int(choices[index, 0].item())
    dx = int(choices[index, 1].item())
    pulse = float(torch.empty((1,)).uniform_(0.08, 0.18, generator=generator).item())
    return dy, dx, pulse


def _make_video(source: torch.Tensor, generator: torch.Generator, num_frames: int) -> torch.Tensor:
    image_size = int(source.shape[-1])
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, image_size, dtype=torch.float32),
        torch.linspace(-1.0, 1.0, image_size, dtype=torch.float32),
        indexing="ij",
    )
    base_glow = (0.5 + 0.5 * torch.cos(xx * 2.5 - yy * 3.0)).unsqueeze(0)
    edge = torch.clamp(source - torch.roll(source, shifts=1, dims=2), min=0.0, max=1.0)
    dy, dx, pulse = _sample_motion(generator)
    positions = torch.linspace(-1.0, 1.0, num_frames, dtype=torch.float32)

    frames: list[torch.Tensor] = []
    for frame_idx, position in enumerate(positions):
        shift_y = int(round(float(position.item()) * dy))
        shift_x = int(round(float(position.item()) * dx))
        shifted = torch.roll(source, shifts=(shift_y, shift_x), dims=(1, 2))
        accent = torch.roll(edge, shifts=(shift_y, shift_x), dims=(1, 2))
        brightness = 0.92 + pulse * torch.cos(torch.tensor(frame_idx * 0.9, dtype=torch.float32))
        frame = brightness * shifted + 0.12 * (1.0 - brightness) * accent + 0.08 * (1.0 - shifted) * base_glow
        frame += 0.015 * torch.randn(source.shape, generator=generator, dtype=torch.float32)
        frame += 0.01 * torch.rand(source.shape, generator=generator, dtype=torch.float32)
        frames.append(frame.clamp(0.0, 1.0))

    return torch.stack(frames, dim=0)


def _make_synthetic_image_to_video_data(cfg: DataConfig) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(int(cfg.seed))
    sources: list[torch.Tensor] = []
    videos: list[torch.Tensor] = []

    for _ in range(int(cfg.num_samples)):
        mask = _shape_mask(int(cfg.image_size), generator)
        texture = _texture(int(cfg.image_size), generator)
        source = _make_source(mask, texture, generator)
        video = _make_video(source, generator, int(cfg.num_frames))
        sources.append(source)
        videos.append(video)

    return torch.stack(sources, dim=0), torch.stack(videos, dim=0)


class SyntheticImageToVideoDataset(Dataset):
    def __init__(self, cfg: DataConfig) -> None:
        self.cfg = cfg
        self._source, self._video = _make_synthetic_image_to_video_data(cfg)

    def __len__(self) -> int:
        return int(self._source.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        i = int(idx)
        return self._source[i], self._video[i]


def get_dataloaders(cfg: DataConfig) -> tuple[DataLoader, DataLoader]:
    dataset: Dataset = SyntheticImageToVideoDataset(cfg)
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


__all__ = ["DataConfig", "SyntheticImageToVideoDataset", "get_dataloaders"]
