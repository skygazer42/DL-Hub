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
    noise_std: float = 0.01
    min_shapes: int = 2
    max_shapes: int = 5
    beta_min: float = 0.8
    beta_max: float = 1.8


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
    if float(cfg.noise_std) < 0.0:
        raise ValueError("noise_std must be >= 0")
    if int(cfg.min_shapes) <= 0 or int(cfg.max_shapes) < int(cfg.min_shapes):
        raise ValueError("invalid shape count range")
    if float(cfg.beta_min) <= 0.0 or float(cfg.beta_max) < float(cfg.beta_min):
        raise ValueError("beta range must satisfy 0 < beta_min <= beta_max")


def _sample_color(generator: torch.Generator) -> torch.Tensor:
    return torch.rand((3,), generator=generator, dtype=torch.float32) * 0.75 + 0.2


def _draw_rectangle(
    image: torch.Tensor,
    depth: torch.Tensor,
    *,
    top: int,
    left: int,
    height: int,
    width: int,
    color: torch.Tensor,
    depth_value: float,
) -> None:
    image[:, top : top + height, left : left + width] = color.view(-1, 1, 1)
    depth[:, top : top + height, left : left + width] = float(depth_value)


def _draw_circle(
    image: torch.Tensor,
    depth: torch.Tensor,
    *,
    cy: int,
    cx: int,
    radius: int,
    color: torch.Tensor,
    depth_value: float,
) -> None:
    h, w = image.shape[-2:]
    yy = torch.arange(h, dtype=torch.float32).view(h, 1)
    xx = torch.arange(w, dtype=torch.float32).view(1, w)
    mask = ((yy - float(cy)) ** 2 + (xx - float(cx)) ** 2) <= float(radius * radius)
    image[:, mask] = color.view(-1, 1)
    depth[:, mask] = float(depth_value)


def make_clean_image_and_depth(cfg: DataConfig, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
    _validate_config(cfg)
    size = int(cfg.image_size)
    seed = int(cfg.seed) * 1_000_003 + int(idx) * 97
    gen = torch.Generator().manual_seed(seed)

    yy = torch.linspace(0.0, 1.0, size, dtype=torch.float32).view(1, size, 1)
    xx = torch.linspace(0.0, 1.0, size, dtype=torch.float32).view(1, 1, size)
    clean = torch.zeros((3, size, size), dtype=torch.float32)
    clean[0] = 0.10 + 0.35 * yy.squeeze(0) + 0.08 * xx.squeeze(0)
    clean[1] = 0.08 + 0.25 * yy.squeeze(0)
    clean[2] = 0.14 + 0.18 * xx.squeeze(0)

    depth = (0.25 + 0.75 * yy).expand(1, size, size).clone()

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
        depth_value = float(torch.rand((), generator=gen).item() * 0.55 + 0.10)
        if shape_idx % 2 == 0:
            rect_h = int(torch.randint(min_size, max_size + 1, (1,), generator=gen).item())
            rect_w = int(torch.randint(min_size, max_size + 1, (1,), generator=gen).item())
            top = int(torch.randint(0, size - rect_h + 1, (1,), generator=gen).item())
            left = int(torch.randint(0, size - rect_w + 1, (1,), generator=gen).item())
            _draw_rectangle(
                clean,
                depth,
                top=top,
                left=left,
                height=rect_h,
                width=rect_w,
                color=color,
                depth_value=depth_value,
            )
        else:
            radius = int(torch.randint(min_size // 2, max_size // 2 + 1, (1,), generator=gen).item())
            cy = int(torch.randint(radius, size - radius + 1, (1,), generator=gen).item())
            cx = int(torch.randint(radius, size - radius + 1, (1,), generator=gen).item())
            _draw_circle(
                clean,
                depth,
                cy=cy,
                cx=cx,
                radius=radius,
                color=color,
                depth_value=depth_value,
            )

    return clean.clamp(0.0, 1.0), depth.clamp(0.0, 1.0)


def synthesize_haze(
    clean: torch.Tensor,
    depth: torch.Tensor,
    *,
    cfg: DataConfig,
    idx: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if clean.ndim != 3 or depth.ndim != 3:
        raise ValueError("Expected clean and depth tensors with shape (C,H,W) and (1,H,W)")
    _validate_config(cfg)

    gen = torch.Generator().manual_seed(int(cfg.seed) * 1_000_003 + int(idx) * 97 + 11)
    beta = float(cfg.beta_min) + float(torch.rand((), generator=gen).item()) * float(
        cfg.beta_max - cfg.beta_min
    )
    transmission = torch.exp(-beta * depth).clamp(0.15, 0.98)
    airlight = (_sample_color(gen) * 0.6 + 0.35).view(3, 1, 1)
    hazy = clean * transmission + airlight * (1.0 - transmission)

    if float(cfg.noise_std) > 0.0:
        noise = torch.randn(hazy.shape, generator=gen, dtype=hazy.dtype) * float(cfg.noise_std)
        hazy = hazy + noise

    return hazy.clamp(0.0, 1.0), transmission.to(torch.float32)


class SyntheticImageDehazingDataset(Dataset):
    def __init__(self, cfg: DataConfig) -> None:
        _validate_config(cfg)
        self.cfg = cfg

    def __len__(self) -> int:
        return int(self.cfg.num_samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        clean, depth = make_clean_image_and_depth(self.cfg, int(idx))
        hazy, transmission = synthesize_haze(clean, depth, cfg=self.cfg, idx=int(idx))
        return hazy, {"clean": clean, "transmission": transmission}


def get_dataloaders(cfg: DataConfig) -> tuple[DataLoader, DataLoader]:
    _validate_config(cfg)
    ds = SyntheticImageDehazingDataset(cfg)
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
    "SyntheticImageDehazingDataset",
    "get_dataloaders",
    "make_clean_image_and_depth",
    "synthesize_haze",
]
