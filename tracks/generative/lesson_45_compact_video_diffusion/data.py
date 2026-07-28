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
    motion_dim: int = 3


def _sample_motion_code(motion_dim: int, generator: torch.Generator) -> torch.Tensor:
    if motion_dim != 3:
        raise ValueError("motion_dim must be 3 for this synthetic lesson")

    angle = float(torch.empty((1,)).uniform_(0.0, 2.0 * torch.pi, generator=generator).item())
    speed = float(torch.empty((1,)).uniform_(0.18, 0.30, generator=generator).item())
    brightness = float(torch.empty((1,)).uniform_(0.55, 0.95, generator=generator).item())
    dx = speed * torch.cos(torch.tensor(angle)).item()
    dy = speed * torch.sin(torch.tensor(angle)).item()
    return torch.tensor([dx, dy, brightness], dtype=torch.float32)


def _render_blob(
    image_size: int,
    center_x: float,
    center_y: float,
    brightness: float,
    generator: torch.Generator,
) -> torch.Tensor:
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, image_size, dtype=torch.float32),
        torch.linspace(-1.0, 1.0, image_size, dtype=torch.float32),
        indexing="ij",
    )
    sigma = 0.16
    dist = (xx - center_x).pow(2) + (yy - center_y).pow(2)
    blob = float(brightness) * torch.exp(-dist / (2.0 * sigma * sigma))
    background = 0.06 + 0.02 * torch.cos(3.0 * xx - 2.0 * yy)
    frame = blob + background
    frame += 0.01 * torch.rand((image_size, image_size), generator=generator, dtype=torch.float32)
    return frame.clamp(0.0, 1.0).unsqueeze(0)


def _make_video(cfg: DataConfig, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    motion_code = _sample_motion_code(int(cfg.motion_dim), generator)
    start_x = float(torch.empty((1,)).uniform_(-0.45, 0.45, generator=generator).item())
    start_y = float(torch.empty((1,)).uniform_(-0.45, 0.45, generator=generator).item())
    dx, dy, brightness = [float(value) for value in motion_code.tolist()]

    frames: list[torch.Tensor] = []
    steps = max(1, int(cfg.num_frames) - 1)
    for frame_idx in range(int(cfg.num_frames)):
        alpha = float(frame_idx) / float(steps)
        center_x = max(-0.8, min(0.8, start_x + alpha * dx))
        center_y = max(-0.8, min(0.8, start_y + alpha * dy))
        frames.append(_render_blob(int(cfg.image_size), center_x, center_y, brightness, generator))

    video = torch.stack(frames, dim=1)
    keyframe = video[:, 0]
    return keyframe, motion_code, video


class SyntheticVideoDiffusionDataset(Dataset):
    def __init__(self, cfg: DataConfig) -> None:
        self.cfg = cfg
        generator = torch.Generator().manual_seed(int(cfg.seed))
        keyframes: list[torch.Tensor] = []
        motion_codes: list[torch.Tensor] = []
        videos: list[torch.Tensor] = []

        for _ in range(int(cfg.num_samples)):
            keyframe, motion_code, video = _make_video(cfg, generator)
            keyframes.append(keyframe)
            motion_codes.append(motion_code)
            videos.append(video)

        self._keyframes = torch.stack(keyframes, dim=0)
        self._motion_codes = torch.stack(motion_codes, dim=0)
        self._videos = torch.stack(videos, dim=0)

    def __len__(self) -> int:
        return int(self._videos.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        i = int(idx)
        return self._keyframes[i], self._motion_codes[i], self._videos[i]


def get_dataloaders(cfg: DataConfig) -> tuple[DataLoader, DataLoader]:
    dataset: Dataset = SyntheticVideoDiffusionDataset(cfg)
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


__all__ = ["DataConfig", "SyntheticVideoDiffusionDataset", "get_dataloaders"]
