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
    num_bins: int = 5
    val_fraction: float = 0.2
    seed: int = 0
    num_workers: int = 0
    noise_std: float = 0.01


def _gaussian_blob(size: int, cx: float, cy: float, sigma: float) -> torch.Tensor:
    yy, xx = torch.meshgrid(
        torch.linspace(0.0, float(size - 1), size, dtype=torch.float32),
        torch.linspace(0.0, float(size - 1), size, dtype=torch.float32),
        indexing="ij",
    )
    dist2 = (xx - cx).pow(2) + (yy - cy).pow(2)
    return torch.exp(-dist2 / max(1e-6, 2.0 * sigma * sigma))


class SyntheticEventCameraDataset(Dataset):
    """Synthetic event volumes with polarity and motion supervision."""

    def __init__(self, cfg: DataConfig) -> None:
        self.cfg = cfg
        if int(cfg.image_size) < 16:
            raise ValueError("image_size must be >= 16")
        if int(cfg.num_bins) < 3:
            raise ValueError("num_bins must be >= 3")
        if not (0.0 < float(cfg.val_fraction) < 1.0):
            raise ValueError("val_fraction must be in (0, 1)")
        if float(cfg.noise_std) < 0.0:
            raise ValueError("noise_std must be >= 0")

    def __len__(self) -> int:
        return int(self.cfg.num_samples)

    def _generator(self, idx: int) -> torch.Generator:
        seed = int(self.cfg.seed) * 1_000_003 + int(idx) * 97 + 80
        return torch.Generator().manual_seed(seed)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        cfg = self.cfg
        idx = int(idx)
        size = int(cfg.image_size)
        bins = int(cfg.num_bins)
        gen = self._generator(idx)

        start_x = float(torch.empty((1,), dtype=torch.float32).uniform_(6.0, float(size - 7), generator=gen).item())
        start_y = float(torch.empty((1,), dtype=torch.float32).uniform_(6.0, float(size - 7), generator=gen).item())
        dx = int(torch.randint(-2, 3, (1,), generator=gen).item())
        dy = int(torch.randint(-2, 3, (1,), generator=gen).item())
        if dx == 0 and dy == 0:
            dx = 1

        sigma = float(torch.empty((1,), dtype=torch.float32).uniform_(2.2, 4.0, generator=gen).item())
        amplitude = float(torch.empty((1,), dtype=torch.float32).uniform_(0.75, 1.1, generator=gen).item())

        event_slices: list[torch.Tensor] = []
        total_pos = torch.zeros((size, size), dtype=torch.float32)
        total_neg = torch.zeros((size, size), dtype=torch.float32)
        prev_intensity: torch.Tensor | None = None

        for bin_idx in range(bins):
            alpha = float(bin_idx) / float(max(1, bins - 1))
            cx = max(2.0, min(float(size - 3), start_x + alpha * float(dx) * 3.0))
            cy = max(2.0, min(float(size - 3), start_y + alpha * float(dy) * 3.0))
            intensity = amplitude * _gaussian_blob(size, cx, cy, sigma)

            if prev_intensity is None:
                signed = torch.zeros_like(intensity)
            else:
                signed = intensity - prev_intensity
                total_pos += torch.clamp(signed, min=0.0)
                total_neg += torch.clamp(-signed, min=0.0)

            event_slices.append(signed)
            prev_intensity = intensity

        event_volume = torch.stack(event_slices, dim=0)
        if float(cfg.noise_std) > 0.0:
            event_volume = event_volume + torch.randn(
                event_volume.shape,
                generator=gen,
                dtype=torch.float32,
            ) * float(cfg.noise_std)
        event_volume = event_volume.to(torch.float32)

        denom = float(max(1, bins - 1))
        motion = torch.stack(
            [
                torch.full((size, size), float(dx) / denom, dtype=torch.float32),
                torch.full((size, size), float(dy) / denom, dtype=torch.float32),
            ],
            dim=0,
        )
        speed = torch.sqrt(motion[0].pow(2) + motion[1].pow(2))
        depth_like = (1.0 / (1.0 + 2.5 * speed)).clamp(0.0, 1.0).unsqueeze(0)

        polarity_map = torch.stack(
            [
                (total_pos / (total_pos.max() + 1e-6)).clamp(0.0, 1.0),
                (total_neg / (total_neg.max() + 1e-6)).clamp(0.0, 1.0),
            ],
            dim=0,
        )
        target = {
            "polarity_map": polarity_map,
            "motion": motion,
            "depth_like": depth_like,
        }
        return event_volume, target


def get_dataloaders(cfg: DataConfig) -> tuple[DataLoader, DataLoader]:
    dataset = SyntheticEventCameraDataset(cfg)
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


__all__ = ["DataConfig", "SyntheticEventCameraDataset", "get_dataloaders"]
