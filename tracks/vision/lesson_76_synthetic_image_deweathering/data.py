from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
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
    min_shapes: int = 2
    max_shapes: int = 5
    streak_count_min: int = 5
    streak_count_max: int = 12
    weather_strength_min: float = 0.12
    weather_strength_max: float = 0.28
    haze_strength_min: float = 0.05
    haze_strength_max: float = 0.20
    snow_blob_count: int = 10


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
    if int(cfg.min_shapes) <= 0 or int(cfg.max_shapes) < int(cfg.min_shapes):
        raise ValueError("invalid shape count range")
    if int(cfg.streak_count_min) <= 0 or int(cfg.streak_count_max) < int(cfg.streak_count_min):
        raise ValueError("invalid streak count range")
    if float(cfg.weather_strength_min) < 0.0 or float(cfg.weather_strength_max) < float(
        cfg.weather_strength_min
    ):
        raise ValueError("invalid weather strength range")
    if float(cfg.haze_strength_min) < 0.0 or float(cfg.haze_strength_max) < float(cfg.haze_strength_min):
        raise ValueError("invalid haze strength range")
    if int(cfg.snow_blob_count) < 0:
        raise ValueError("snow_blob_count must be >= 0")


def _sample_color(generator: torch.Generator) -> torch.Tensor:
    return torch.rand((3,), generator=generator, dtype=torch.float32) * 0.7 + 0.15


def _draw_rectangle(
    image: torch.Tensor,
    *,
    top: int,
    left: int,
    height: int,
    width: int,
    color: torch.Tensor,
) -> None:
    image[:, top : top + height, left : left + width] = color.view(-1, 1, 1)


def _draw_circle(
    image: torch.Tensor,
    *,
    cy: int,
    cx: int,
    radius: int,
    color: torch.Tensor,
) -> None:
    h, w = image.shape[-2:]
    yy = torch.arange(h, dtype=torch.float32).view(h, 1)
    xx = torch.arange(w, dtype=torch.float32).view(1, w)
    mask = ((yy - float(cy)) ** 2 + (xx - float(cx)) ** 2) <= float(radius * radius)
    image[:, mask] = color.view(-1, 1)


def make_clean_image(cfg: DataConfig, idx: int) -> torch.Tensor:
    _validate_config(cfg)
    size = int(cfg.image_size)
    gen = torch.Generator().manual_seed(int(cfg.seed) * 1_000_003 + int(idx) * 97)

    yy = torch.linspace(0.0, 1.0, size, dtype=torch.float32).view(size, 1)
    xx = torch.linspace(0.0, 1.0, size, dtype=torch.float32).view(1, size)
    clean = torch.zeros((3, size, size), dtype=torch.float32)
    clean[0] = 0.12 + 0.24 * yy + 0.08 * xx
    clean[1] = 0.10 + 0.28 * yy
    clean[2] = 0.14 + 0.22 * xx

    num_shapes = int(
        torch.randint(
            low=int(cfg.min_shapes),
            high=int(cfg.max_shapes) + 1,
            size=(1,),
            generator=gen,
        ).item()
    )
    min_size = max(4, size // 8)
    max_size = max(min_size + 1, size // 2)
    for shape_idx in range(num_shapes):
        color = _sample_color(gen)
        if shape_idx % 2 == 0:
            rect_h = int(torch.randint(min_size, max_size + 1, (1,), generator=gen).item())
            rect_w = int(torch.randint(min_size, max_size + 1, (1,), generator=gen).item())
            top = int(torch.randint(0, size - rect_h + 1, (1,), generator=gen).item())
            left = int(torch.randint(0, size - rect_w + 1, (1,), generator=gen).item())
            _draw_rectangle(clean, top=top, left=left, height=rect_h, width=rect_w, color=color)
        else:
            radius = int(torch.randint(min_size // 2, max_size // 2 + 1, (1,), generator=gen).item())
            cy = int(torch.randint(radius, size - radius + 1, (1,), generator=gen).item())
            cx = int(torch.randint(radius, size - radius + 1, (1,), generator=gen).item())
            _draw_circle(clean, cy=cy, cx=cx, radius=radius, color=color)
    return clean.clamp(0.0, 1.0)


def make_weather_residual(cfg: DataConfig, idx: int) -> torch.Tensor:
    _validate_config(cfg)
    size = int(cfg.image_size)
    gen = torch.Generator().manual_seed(int(cfg.seed) * 1_000_003 + int(idx) * 97 + 41)

    residual = torch.zeros((3, size, size), dtype=torch.float32)
    haze_strength = float(cfg.haze_strength_min) + float(torch.rand((), generator=gen).item()) * float(
        cfg.haze_strength_max - cfg.haze_strength_min
    )
    yy = torch.linspace(0.0, 1.0, size, dtype=torch.float32).view(1, size, 1)
    xx = torch.linspace(0.0, 1.0, size, dtype=torch.float32).view(1, 1, size)
    haze = (0.6 * yy + 0.4 * xx).expand(3, size, size)
    residual = residual + haze * haze_strength

    streak_count = int(
        torch.randint(
            int(cfg.streak_count_min),
            int(cfg.streak_count_max) + 1,
            (1,),
            generator=gen,
        ).item()
    )
    for _ in range(streak_count):
        length = int(torch.randint(max(4, size // 6), max(5, size // 3), (1,), generator=gen).item())
        start_y = int(torch.randint(0, max(1, size - length + 1), (1,), generator=gen).item())
        start_x = int(torch.randint(0, size, (1,), generator=gen).item())
        shift = int(torch.randint(-1, 2, (1,), generator=gen).item())
        strength = float(cfg.weather_strength_min) + float(torch.rand((), generator=gen).item()) * float(
            cfg.weather_strength_max - cfg.weather_strength_min
        )
        for step in range(length):
            y = start_y + step
            x = start_x + shift * step
            if 0 <= y < size and 0 <= x < size:
                residual[:, y, x] += torch.tensor(
                    [0.75 * strength, 0.82 * strength, strength],
                    dtype=torch.float32,
                )
                if x + 1 < size:
                    residual[:, y, x + 1] += torch.tensor(
                        [0.18 * strength, 0.22 * strength, 0.26 * strength],
                        dtype=torch.float32,
                    )

    for _ in range(int(cfg.snow_blob_count)):
        radius = int(torch.randint(1, max(2, size // 10), (1,), generator=gen).item())
        cy = int(torch.randint(radius, size - radius + 1, (1,), generator=gen).item())
        cx = int(torch.randint(radius, size - radius + 1, (1,), generator=gen).item())
        strength = float(cfg.weather_strength_min) + float(torch.rand((), generator=gen).item()) * float(
            cfg.weather_strength_max - cfg.weather_strength_min
        )
        h, w = residual.shape[-2:]
        yy_grid = torch.arange(h, dtype=torch.float32).view(h, 1)
        xx_grid = torch.arange(w, dtype=torch.float32).view(1, w)
        mask = ((yy_grid - float(cy)) ** 2 + (xx_grid - float(cx)) ** 2) <= float(radius * radius)
        residual[:, mask] += torch.tensor([strength, strength, strength], dtype=torch.float32).view(-1, 1)

    residual = F.avg_pool2d(residual.unsqueeze(0), kernel_size=3, stride=1, padding=1).squeeze(0)
    return residual.clamp(0.0, 0.55)


def synthesize_weathered_image(
    clean: torch.Tensor,
    weather_residual: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if clean.ndim != 3 or weather_residual.ndim != 3:
        raise ValueError("Expected clean and weather_residual tensors with shape (C,H,W)")
    weathered = (clean + weather_residual).clamp(0.0, 1.0)
    residual = weathered - clean
    return weathered, residual.to(torch.float32)


class SyntheticImageDeweatheringDataset(Dataset):
    def __init__(self, cfg: DataConfig) -> None:
        _validate_config(cfg)
        self.cfg = cfg

    def __len__(self) -> int:
        return int(self.cfg.num_samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        clean = make_clean_image(self.cfg, int(idx))
        residual = make_weather_residual(self.cfg, int(idx))
        weathered, residual = synthesize_weathered_image(clean, residual)
        return weathered, {"clean": clean, "weather_residual": residual}


def get_dataloaders(cfg: DataConfig) -> tuple[DataLoader, DataLoader]:
    _validate_config(cfg)
    ds = SyntheticImageDeweatheringDataset(cfg)
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
    "SyntheticImageDeweatheringDataset",
    "get_dataloaders",
    "make_clean_image",
    "make_weather_residual",
    "synthesize_weathered_image",
]
