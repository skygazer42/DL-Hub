from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from dlhub.data.splits import train_val_split_indices


@dataclass(frozen=True)
class DataConfig:
    num_samples: int = 1024
    batch_size: int = 8
    frames: int = 5
    image_size: int = 40
    val_fraction: float = 0.2
    seed: int = 0
    num_workers: int = 0
    in_channels: int = 1
    noise_std: float = 0.08


class SyntheticVideoRestorationDataset(Dataset):
    """Generate paired synthetic clips: degraded input and clean restoration target."""

    def __init__(self, cfg: DataConfig) -> None:
        self.cfg = cfg
        if int(cfg.frames) < 3:
            raise ValueError("frames must be >= 3")
        if int(cfg.image_size) < 24:
            raise ValueError("image_size must be >= 24")
        if not (0.0 < float(cfg.val_fraction) < 1.0):
            raise ValueError("val_fraction must be in (0, 1)")
        if int(cfg.in_channels) < 1:
            raise ValueError("in_channels must be >= 1")
        if float(cfg.noise_std) < 0.0:
            raise ValueError("noise_std must be >= 0")

    def __len__(self) -> int:
        return int(self.cfg.num_samples)

    def _generator(self, idx: int) -> torch.Generator:
        seed = int(self.cfg.seed) * 1_000_003 + int(idx) * 193 + 41
        return torch.Generator().manual_seed(seed)

    def _build_clean_clip(self, gen: torch.Generator) -> torch.Tensor:
        cfg = self.cfg
        t = int(cfg.frames)
        c = int(cfg.in_channels)
        size = int(cfg.image_size)
        yy, xx = torch.meshgrid(
            torch.arange(size, dtype=torch.float32),
            torch.arange(size, dtype=torch.float32),
            indexing="ij",
        )
        x_norm = xx / max(1.0, float(size - 1))
        y_norm = yy / max(1.0, float(size - 1))

        background = 0.10 + 0.20 * (1.0 - y_norm)
        clip = torch.zeros((t, c, size, size), dtype=torch.float32)

        base_x = float(torch.randint(int(size * 0.22), int(size * 0.40), (1,), generator=gen).item())
        base_y = float(torch.randint(int(size * 0.28), int(size * 0.72), (1,), generator=gen).item())
        velocity = float(torch.randint(1, 3, (1,), generator=gen).item())
        sigma = max(2.0, float(size) / 10.0)

        for frame_idx in range(t):
            cx = base_x + velocity * frame_idx
            cy = base_y + 1.5 * torch.sin(torch.tensor(float(frame_idx) * 0.8)).item()
            blob = torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma * sigma))
            frame = (background + 0.70 * blob).clamp(0.0, 1.0)
            for channel_idx in range(c):
                channel_gain = 1.0 - 0.08 * channel_idx
                clip[frame_idx, channel_idx] = (frame * channel_gain).clamp(0.0, 1.0)
        return clip

    def _degrade_clip(self, clean: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
        cfg = self.cfg
        t, c, h, w = clean.shape
        blurred = clean.clone()
        kernel = torch.tensor(
            [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]],
            dtype=torch.float32,
        )
        kernel = (kernel / kernel.sum()).view(1, 1, 3, 3).repeat(c, 1, 1, 1)
        for frame_idx in range(t):
            frame = blurred[frame_idx].unsqueeze(0)
            frame = F.conv2d(frame, kernel, padding=1, groups=c)
            shift_y = int(torch.randint(-1, 2, (1,), generator=gen).item())
            shift_x = int(torch.randint(-1, 2, (1,), generator=gen).item())
            frame = torch.roll(frame, shifts=(shift_y, shift_x), dims=(-2, -1))
            noise = torch.randn((1, c, h, w), generator=gen, dtype=torch.float32) * float(cfg.noise_std)
            blurred[frame_idx] = (frame + noise).squeeze(0).clamp(0.0, 1.0)
        return blurred

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        gen = self._generator(int(idx))
        clean = self._build_clean_clip(gen)
        degraded = self._degrade_clip(clean, gen)
        return degraded, clean


def get_dataloaders(cfg: DataConfig) -> tuple[DataLoader, DataLoader]:
    ds = SyntheticVideoRestorationDataset(cfg)
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


__all__ = ["DataConfig", "SyntheticVideoRestorationDataset", "get_dataloaders"]
