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
    reflection_strength_min: float = 0.15
    reflection_strength_max: float = 0.55
    blur_kernel_size: int = 3
    min_shapes: int = 2
    max_shapes: int = 5


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
    if float(cfg.reflection_strength_min) < 0.0 or float(cfg.reflection_strength_max) > 1.0:
        raise ValueError("reflection strength must be in [0, 1]")
    if float(cfg.reflection_strength_max) < float(cfg.reflection_strength_min):
        raise ValueError("reflection_strength_max must be >= reflection_strength_min")
    if int(cfg.blur_kernel_size) <= 0 or int(cfg.blur_kernel_size) % 2 == 0:
        raise ValueError("blur_kernel_size must be positive and odd")
    if int(cfg.min_shapes) <= 0 or int(cfg.max_shapes) < int(cfg.min_shapes):
        raise ValueError("invalid shape count range")


def _sample_color(generator: torch.Generator) -> torch.Tensor:
    return torch.rand((3,), generator=generator, dtype=torch.float32) * 0.75 + 0.2


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


def make_transmission_image(cfg: DataConfig, idx: int) -> torch.Tensor:
    _validate_config(cfg)
    size = int(cfg.image_size)
    seed = int(cfg.seed) * 1_000_003 + int(idx) * 97
    gen = torch.Generator().manual_seed(seed)

    yy = torch.linspace(0.0, 1.0, size, dtype=torch.float32).view(size, 1)
    xx = torch.linspace(0.0, 1.0, size, dtype=torch.float32).view(1, size)
    image = torch.zeros((3, size, size), dtype=torch.float32)
    image[0] = 0.08 + 0.32 * yy + 0.18 * xx
    image[1] = 0.12 + 0.45 * yy
    image[2] = 0.10 + 0.40 * xx

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
                image,
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
            _draw_circle(image, cy=cy, cx=cx, radius=radius, color=color)
    return image.clamp(0.0, 1.0)


def make_reflection_layer(cfg: DataConfig, idx: int) -> torch.Tensor:
    _validate_config(cfg)
    size = int(cfg.image_size)
    seed = int(cfg.seed) * 1_000_003 + int(idx) * 97 + 41
    gen = torch.Generator().manual_seed(seed)

    reflection = torch.zeros((3, size, size), dtype=torch.float32)
    num_shapes = int(torch.randint(2, 6, (1,), generator=gen).item())
    min_size = max(3, size // 10)
    max_size = max(min_size + 1, size // 3)
    for shape_idx in range(num_shapes):
        color = _sample_color(gen)
        if shape_idx % 2 == 0:
            rect_h = int(torch.randint(min_size, max_size + 1, (1,), generator=gen).item())
            rect_w = int(torch.randint(min_size, max_size + 1, (1,), generator=gen).item())
            top = int(torch.randint(0, size - rect_h + 1, (1,), generator=gen).item())
            left = int(torch.randint(0, size - rect_w + 1, (1,), generator=gen).item())
            _draw_rectangle(
                reflection,
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
            _draw_circle(reflection, cy=cy, cx=cx, radius=radius, color=color)

    k = int(cfg.blur_kernel_size)
    reflection = torch.nn.functional.avg_pool2d(
        reflection.unsqueeze(0),
        kernel_size=k,
        stride=1,
        padding=k // 2,
    ).squeeze(0)
    brightness = torch.rand((1,), generator=gen, dtype=torch.float32).item() * 0.35 + 0.5
    return (reflection * float(brightness)).clamp(0.0, 1.0)


def synthesize_reflection_mixture(
    transmission: torch.Tensor,
    reflection: torch.Tensor,
    *,
    cfg: DataConfig,
    idx: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    _validate_config(cfg)
    if transmission.ndim != 3 or reflection.ndim != 3:
        raise ValueError("Expected transmission and reflection tensors with shape (C,H,W)")
    gen = torch.Generator().manual_seed(int(cfg.seed) * 1_000_003 + int(idx) * 97 + 83)
    alpha = float(cfg.reflection_strength_min) + float(torch.rand((), generator=gen).item()) * float(
        cfg.reflection_strength_max - cfg.reflection_strength_min
    )
    alpha_map = torch.full((1, transmission.shape[-2], transmission.shape[-1]), alpha, dtype=torch.float32)
    mixture = transmission * (1.0 - alpha_map) + reflection * alpha_map
    return mixture.clamp(0.0, 1.0), alpha_map


class SyntheticReflectionRemovalDataset(Dataset):
    def __init__(self, cfg: DataConfig) -> None:
        _validate_config(cfg)
        self.cfg = cfg

    def __len__(self) -> int:
        return int(self.cfg.num_samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        transmission = make_transmission_image(self.cfg, int(idx))
        reflection = make_reflection_layer(self.cfg, int(idx))
        mixture, _ = synthesize_reflection_mixture(transmission, reflection, cfg=self.cfg, idx=int(idx))
        return mixture, {"transmission": transmission, "reflection": reflection}


def get_dataloaders(cfg: DataConfig) -> tuple[DataLoader, DataLoader]:
    _validate_config(cfg)
    ds = SyntheticReflectionRemovalDataset(cfg)
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
    "SyntheticReflectionRemovalDataset",
    "get_dataloaders",
    "make_reflection_layer",
    "make_transmission_image",
    "synthesize_reflection_mixture",
]
