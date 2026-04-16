from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from dlhub.data.splits import train_val_split_indices


@dataclass(frozen=True)
class DataConfig:
    num_samples: int = 1024
    batch_size: int = 16
    image_size: int = 32
    val_fraction: float = 0.2
    seed: int = 0
    num_workers: int = 0

    in_channels: int = 3
    motion_pixels: int = 3
    noise_std: float = 0.01


class SyntheticVideoFrameInterpolationDataset(Dataset):
    """Synthetic 3-frame clips; predict frame t=1 from frames t=0 and t=2."""

    def __init__(self, cfg: DataConfig) -> None:
        self.cfg = cfg
        if int(cfg.image_size) < 16:
            raise ValueError("image_size must be >= 16")
        if not (0.0 < float(cfg.val_fraction) < 1.0):
            raise ValueError("val_fraction must be in (0, 1)")
        if int(cfg.in_channels) < 1:
            raise ValueError("in_channels must be >= 1")
        if int(cfg.motion_pixels) < 1:
            raise ValueError("motion_pixels must be >= 1")
        if float(cfg.noise_std) < 0.0:
            raise ValueError("noise_std must be >= 0")

    def __len__(self) -> int:
        return int(self.cfg.num_samples)

    def _generator(self, idx: int) -> torch.Generator:
        seed = int(self.cfg.seed) * 1_000_003 + int(idx) * 313 + 17
        return torch.Generator().manual_seed(seed)

    def _render_frame(
        self,
        *,
        size: int,
        channels: int,
        center_x: int,
        center_y: int,
        obj_half: int,
        color: torch.Tensor,
        gen: torch.Generator,
    ) -> torch.Tensor:
        yy, xx = torch.meshgrid(
            torch.arange(size, dtype=torch.float32),
            torch.arange(size, dtype=torch.float32),
            indexing="ij",
        )
        background = (0.15 + 0.35 * (yy / max(1.0, float(size - 1)))).unsqueeze(0).repeat(channels, 1, 1)
        if float(self.cfg.noise_std) > 0.0:
            background = background + float(self.cfg.noise_std) * torch.randn(
                (channels, size, size), generator=gen, dtype=torch.float32
            )

        x1 = max(0, int(center_x - obj_half))
        y1 = max(0, int(center_y - obj_half))
        x2 = min(size, int(center_x + obj_half + 1))
        y2 = min(size, int(center_y + obj_half + 1))
        frame = background.clamp(0.0, 1.0)
        frame[:, y1:y2, x1:x2] = color[:, None, None]
        return frame.clamp(0.0, 1.0)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = self.cfg
        size = int(cfg.image_size)
        channels = int(cfg.in_channels)
        motion = int(cfg.motion_pixels)
        gen = self._generator(int(idx))

        obj_half = max(2, size // 10)
        margin = obj_half + motion + 2
        start_x = int(torch.randint(margin, size - margin, (1,), generator=gen).item())
        start_y = int(torch.randint(margin, size - margin, (1,), generator=gen).item())

        horizontal = bool(torch.randint(0, 2, (1,), generator=gen).item())
        direction = -1 if bool(torch.randint(0, 2, (1,), generator=gen).item()) else 1
        dx = direction * motion if horizontal else 0
        dy = direction * motion if not horizontal else 0

        color = torch.rand((channels,), generator=gen, dtype=torch.float32) * 0.6 + 0.4
        f0 = self._render_frame(
            size=size,
            channels=channels,
            center_x=start_x,
            center_y=start_y,
            obj_half=obj_half,
            color=color,
            gen=gen,
        )
        f1 = self._render_frame(
            size=size,
            channels=channels,
            center_x=start_x + dx,
            center_y=start_y + dy,
            obj_half=obj_half,
            color=color,
            gen=gen,
        )
        f2 = self._render_frame(
            size=size,
            channels=channels,
            center_x=start_x + 2 * dx,
            center_y=start_y + 2 * dy,
            obj_half=obj_half,
            color=color,
            gen=gen,
        )

        endpoints = torch.stack([f0, f2], dim=0)
        return endpoints.to(torch.float32), f1.to(torch.float32)


def get_dataloaders(cfg: DataConfig) -> tuple[DataLoader, DataLoader]:
    ds = SyntheticVideoFrameInterpolationDataset(cfg)
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


__all__ = ["DataConfig", "SyntheticVideoFrameInterpolationDataset", "get_dataloaders"]
