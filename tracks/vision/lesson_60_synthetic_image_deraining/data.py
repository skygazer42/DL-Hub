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
    min_shapes: int = 2
    max_shapes: int = 5
    rain_lines_min: int = 8
    rain_lines_max: int = 18
    rain_length_min: int = 6
    rain_length_max: int = 12
    rain_strength_min: float = 0.10
    rain_strength_max: float = 0.30


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
    if int(cfg.rain_lines_min) <= 0 or int(cfg.rain_lines_max) < int(cfg.rain_lines_min):
        raise ValueError("invalid rain line range")
    if int(cfg.rain_length_min) <= 0 or int(cfg.rain_length_max) < int(cfg.rain_length_min):
        raise ValueError("invalid rain length range")
    if float(cfg.rain_strength_min) < 0.0 or float(cfg.rain_strength_max) < float(
        cfg.rain_strength_min
    ):
        raise ValueError("invalid rain strength range")


def _sample_color(generator: torch.Generator) -> torch.Tensor:
    return torch.rand((3,), generator=generator, dtype=torch.float32) * 0.65 + 0.15


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
    seed = int(cfg.seed) * 1_000_003 + int(idx) * 97
    gen = torch.Generator().manual_seed(seed)

    yy = torch.linspace(0.0, 1.0, size, dtype=torch.float32).view(size, 1)
    xx = torch.linspace(0.0, 1.0, size, dtype=torch.float32).view(1, size)
    clean = torch.zeros((3, size, size), dtype=torch.float32)
    clean[0] = 0.08 + 0.20 * yy + 0.06 * xx
    clean[1] = 0.10 + 0.24 * yy
    clean[2] = 0.12 + 0.20 * xx

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
            _draw_rectangle(
                clean,
                top=top,
                left=left,
                height=rect_h,
                width=rect_w,
                color=color,
            )
        else:
            radius = int(torch.randint(min_size // 2, max_size // 2 + 1, (1,), generator=gen).item())
            cy = int(torch.randint(radius, size - radius + 1, (1,), generator=gen).item())
            cx = int(torch.randint(radius, size - radius + 1, (1,), generator=gen).item())
            _draw_circle(clean, cy=cy, cx=cx, radius=radius, color=color)
    return clean.clamp(0.0, 0.82)


def make_rain_layer(cfg: DataConfig, idx: int) -> torch.Tensor:
    _validate_config(cfg)
    size = int(cfg.image_size)
    gen = torch.Generator().manual_seed(int(cfg.seed) * 1_000_003 + int(idx) * 97 + 41)

    rain = torch.zeros((3, size, size), dtype=torch.float32)
    num_lines = int(
        torch.randint(
            low=int(cfg.rain_lines_min),
            high=int(cfg.rain_lines_max) + 1,
            size=(1,),
            generator=gen,
        ).item()
    )

    for _ in range(num_lines):
        length = int(
            torch.randint(
                low=int(cfg.rain_length_min),
                high=int(cfg.rain_length_max) + 1,
                size=(1,),
                generator=gen,
            ).item()
        )
        start_y = int(torch.randint(0, max(1, size - length + 1), (1,), generator=gen).item())
        start_x = int(torch.randint(0, size, (1,), generator=gen).item())
        shift = int(torch.randint(-1, 2, (1,), generator=gen).item())
        strength = float(cfg.rain_strength_min) + float(torch.rand((), generator=gen).item()) * float(
            cfg.rain_strength_max - cfg.rain_strength_min
        )

        for step in range(length):
            y = start_y + step
            x = start_x + shift * step
            if 0 <= y < size and 0 <= x < size:
                rain[:, y, x] += torch.tensor(
                    [strength * 0.9, strength * 0.95, strength],
                    dtype=torch.float32,
                )
                if x + 1 < size:
                    rain[:, y, x + 1] += torch.tensor(
                        [strength * 0.2, strength * 0.22, strength * 0.25],
                        dtype=torch.float32,
                    )

    rain = torch.nn.functional.avg_pool2d(rain.unsqueeze(0), kernel_size=3, stride=1, padding=1).squeeze(0)
    return rain.clamp(0.0, 1.0)


def synthesize_rainy_image(
    clean: torch.Tensor,
    rain_layer: torch.Tensor,
    *,
    cfg: DataConfig,
) -> torch.Tensor:
    _validate_config(cfg)
    if clean.ndim != 3 or rain_layer.ndim != 3:
        raise ValueError("Expected clean and rain_layer tensors with shape (C,H,W)")
    return (clean + rain_layer).clamp(0.0, 1.0)


class SyntheticImageDerainingDataset(Dataset):
    def __init__(self, cfg: DataConfig) -> None:
        _validate_config(cfg)
        self.cfg = cfg

    def __len__(self) -> int:
        return int(self.cfg.num_samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        clean = make_clean_image(self.cfg, int(idx))
        rain_layer = make_rain_layer(self.cfg, int(idx))
        rainy = synthesize_rainy_image(clean, rain_layer, cfg=self.cfg)
        return rainy, {"clean": clean, "rain_layer": rain_layer}


def get_dataloaders(cfg: DataConfig) -> tuple[DataLoader, DataLoader]:
    _validate_config(cfg)
    ds = SyntheticImageDerainingDataset(cfg)
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
    "SyntheticImageDerainingDataset",
    "get_dataloaders",
    "make_clean_image",
    "make_rain_layer",
    "synthesize_rainy_image",
]
