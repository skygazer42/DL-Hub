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
    anomaly_fraction: float = 0.4
    noise_std: float = 0.01


def _validate_config(cfg: DataConfig) -> None:
    if int(cfg.num_samples) <= 0:
        raise ValueError("num_samples must be > 0")
    if int(cfg.batch_size) <= 0:
        raise ValueError("batch_size must be > 0")
    if int(cfg.image_size) < 16:
        raise ValueError("image_size must be >= 16")
    if not (0.0 < float(cfg.val_fraction) < 1.0):
        raise ValueError("val_fraction must be in (0, 1)")
    if int(cfg.in_channels) != 3:
        raise ValueError("in_channels must be 3 for v1")
    if not (0.0 <= float(cfg.anomaly_fraction) <= 1.0):
        raise ValueError("anomaly_fraction must be in [0, 1]")
    if float(cfg.noise_std) < 0.0:
        raise ValueError("noise_std must be >= 0")


def _base_image(*, size: int, g: torch.Generator) -> torch.Tensor:
    yy = torch.linspace(0.0, 1.0, size, dtype=torch.float32).view(size, 1)
    xx = torch.linspace(0.0, 1.0, size, dtype=torch.float32).view(1, size)
    image = torch.zeros((3, size, size), dtype=torch.float32)
    image[0] = 0.12 + 0.40 * yy + 0.08 * xx
    image[1] = 0.14 + 0.25 * yy + 0.22 * xx
    image[2] = 0.10 + 0.15 * yy + 0.32 * xx

    shape_count = int(torch.randint(2, 5, (1,), generator=g).item())
    for _ in range(shape_count):
        h = int(torch.randint(max(3, size // 8), max(5, size // 3), (1,), generator=g).item())
        w = int(torch.randint(max(3, size // 8), max(5, size // 3), (1,), generator=g).item())
        top = int(torch.randint(0, size - h + 1, (1,), generator=g).item())
        left = int(torch.randint(0, size - w + 1, (1,), generator=g).item())
        color = torch.rand((3, 1, 1), generator=g, dtype=torch.float32) * 0.7 + 0.2
        image[:, top : top + h, left : left + w] = color
    return image.clamp(0.0, 1.0)


def _inject_anomaly(
    image: torch.Tensor,
    *,
    size: int,
    g: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    out = image.clone()
    anomaly_map = torch.zeros((1, size, size), dtype=torch.float32)
    patch_h = int(torch.randint(max(3, size // 10), max(6, size // 4), (1,), generator=g).item())
    patch_w = int(torch.randint(max(3, size // 10), max(6, size // 4), (1,), generator=g).item())
    top = int(torch.randint(0, size - patch_h + 1, (1,), generator=g).item())
    left = int(torch.randint(0, size - patch_w + 1, (1,), generator=g).item())

    anomaly_pattern = torch.rand((3, patch_h, patch_w), generator=g, dtype=torch.float32)
    out[:, top : top + patch_h, left : left + patch_w] = anomaly_pattern
    anomaly_map[:, top : top + patch_h, left : left + patch_w] = 1.0
    anomaly_map = torch.nn.functional.avg_pool2d(
        anomaly_map.unsqueeze(0), kernel_size=3, stride=1, padding=1
    ).squeeze(0)
    return out.clamp(0.0, 1.0), anomaly_map.clamp(0.0, 1.0)


class SyntheticVisionAnomalyDataset(Dataset):
    def __init__(self, cfg: DataConfig) -> None:
        _validate_config(cfg)
        self.cfg = cfg

    def __len__(self) -> int:
        return int(self.cfg.num_samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        sample_idx = int(idx)
        cfg = self.cfg
        size = int(cfg.image_size)
        g = torch.Generator().manual_seed(int(cfg.seed) * 1_000_003 + sample_idx * 97)

        clean = _base_image(size=size, g=g)
        image = clean.clone()
        anomaly_map = torch.zeros((1, size, size), dtype=torch.float32)
        label = torch.tensor(0.0, dtype=torch.float32)
        if float(torch.rand((), generator=g).item()) < float(cfg.anomaly_fraction):
            image, anomaly_map = _inject_anomaly(clean, size=size, g=g)
            label = torch.tensor(1.0, dtype=torch.float32)

        if float(cfg.noise_std) > 0.0:
            noise = torch.randn(image.shape, generator=g, dtype=torch.float32) * float(cfg.noise_std)
            image = (image + noise).clamp(0.0, 1.0)

        return image.to(torch.float32), {
            "reconstruction": clean.to(torch.float32),
            "anomaly_map": anomaly_map.to(torch.float32),
            "label": label.to(torch.float32),
        }


def get_dataloaders(cfg: DataConfig) -> tuple[DataLoader, DataLoader]:
    _validate_config(cfg)
    ds = SyntheticVisionAnomalyDataset(cfg)
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


__all__ = ["DataConfig", "SyntheticVisionAnomalyDataset", "get_dataloaders"]

