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
    glass_strength_min: float = 0.20
    glass_strength_max: float = 0.50
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
    if int(cfg.min_shapes) <= 0 or int(cfg.max_shapes) < int(cfg.min_shapes):
        raise ValueError("invalid shape count range")
    if float(cfg.glass_strength_min) < 0.0 or float(cfg.glass_strength_max) < float(
        cfg.glass_strength_min
    ):
        raise ValueError("invalid glass strength range")
    if float(cfg.noise_std) < 0.0:
        raise ValueError("noise_std must be >= 0")


def _sample_color(generator: torch.Generator) -> torch.Tensor:
    return torch.rand((3,), generator=generator, dtype=torch.float32) * 0.7 + 0.15


def make_scene(cfg: DataConfig, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
    _validate_config(cfg)
    size = int(cfg.image_size)
    gen = torch.Generator().manual_seed(int(cfg.seed) * 1_000_003 + int(idx) * 97)

    yy = torch.linspace(0.0, 1.0, size, dtype=torch.float32).view(size, 1)
    xx = torch.linspace(0.0, 1.0, size, dtype=torch.float32).view(1, size)
    image = torch.zeros((3, size, size), dtype=torch.float32)
    image[0] = 0.08 + 0.24 * yy + 0.06 * xx
    image[1] = 0.10 + 0.28 * yy
    image[2] = 0.14 + 0.20 * xx
    depth = (0.28 + 0.65 * yy).expand(1, size, size).clone()

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
    yy_grid = torch.arange(size, dtype=torch.float32).view(size, 1)
    xx_grid = torch.arange(size, dtype=torch.float32).view(1, size)
    for shape_idx in range(num_shapes):
        color = _sample_color(gen)
        depth_value = float(torch.rand((), generator=gen).item() * 0.55 + 0.10)
        if shape_idx % 2 == 0:
            rect_h = int(torch.randint(min_size, max_size + 1, (1,), generator=gen).item())
            rect_w = int(torch.randint(min_size, max_size + 1, (1,), generator=gen).item())
            top = int(torch.randint(0, size - rect_h + 1, (1,), generator=gen).item())
            left = int(torch.randint(0, size - rect_w + 1, (1,), generator=gen).item())
            image[:, top : top + rect_h, left : left + rect_w] = color.view(-1, 1, 1)
            depth[:, top : top + rect_h, left : left + rect_w] = depth_value
        else:
            radius_x = int(torch.randint(min_size // 2, max_size // 2 + 1, (1,), generator=gen).item())
            radius_y = int(torch.randint(min_size // 2, max_size // 2 + 1, (1,), generator=gen).item())
            cy = int(torch.randint(radius_y, size - radius_y + 1, (1,), generator=gen).item())
            cx = int(torch.randint(radius_x, size - radius_x + 1, (1,), generator=gen).item())
            mask = (
                ((yy_grid - float(cy)) / max(1.0, float(radius_y))) ** 2
                + ((xx_grid - float(cx)) / max(1.0, float(radius_x))) ** 2
            ) <= 1.0
            image[:, mask] = color.view(-1, 1)
            depth[:, mask] = depth_value

    return image.clamp(0.0, 1.0), depth.clamp(0.05, 1.0)


def add_transparent_overlay(
    image: torch.Tensor,
    depth: torch.Tensor,
    *,
    cfg: DataConfig,
    idx: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if image.ndim != 3 or depth.ndim != 3:
        raise ValueError("Expected image and depth tensors with shape (C,H,W) and (1,H,W)")
    _validate_config(cfg)

    size = int(cfg.image_size)
    gen = torch.Generator().manual_seed(int(cfg.seed) * 1_000_003 + int(idx) * 97 + 53)
    yy_grid = torch.arange(size, dtype=torch.float32).view(size, 1)
    xx_grid = torch.arange(size, dtype=torch.float32).view(1, size)

    observed = image.clone()
    target_depth = depth.clone()
    transparency = torch.zeros((1, size, size), dtype=torch.float32)

    num_glass = int(torch.randint(1, 4, (1,), generator=gen).item())
    min_size = max(5, size // 7)
    max_size = max(min_size + 1, size // 3)
    for glass_idx in range(num_glass):
        radius_x = int(torch.randint(min_size, max_size + 1, (1,), generator=gen).item())
        radius_y = int(torch.randint(min_size, max_size + 1, (1,), generator=gen).item())
        cy = int(torch.randint(radius_y, size - radius_y + 1, (1,), generator=gen).item())
        cx = int(torch.randint(radius_x, size - radius_x + 1, (1,), generator=gen).item())
        mask = (
            ((yy_grid - float(cy)) / max(1.0, float(radius_y))) ** 2
            + ((xx_grid - float(cx)) / max(1.0, float(radius_x))) ** 2
        ) <= 1.0
        strength = float(cfg.glass_strength_min) + float(torch.rand((), generator=gen).item()) * float(
            cfg.glass_strength_max - cfg.glass_strength_min
        )
        glass_depth = float(torch.rand((), generator=gen).item() * 0.45 + 0.12)
        stripe = 0.5 + 0.5 * torch.sin((xx_grid + yy_grid) / max(2.0, float(3 + glass_idx)))
        tint = torch.stack(
            (
                0.55 + 0.25 * stripe,
                0.62 + 0.20 * stripe,
                0.70 + 0.18 * stripe,
            ),
            dim=0,
        ).to(torch.float32)

        if bool(mask.any()):
            observed[:, mask] = observed[:, mask] * (1.0 - strength) + tint[:, mask] * strength
            target_depth[:, mask] = glass_depth
            transparency[:, mask] = 1.0

    if float(cfg.noise_std) > 0.0:
        observed = observed + torch.randn(observed.shape, generator=gen, dtype=torch.float32) * float(
            cfg.noise_std
        )

    return observed.clamp(0.0, 1.0), {
        "depth": target_depth.clamp(0.05, 1.0),
        "transparency": transparency,
    }


class SyntheticTransparentDepthDataset(Dataset):
    def __init__(self, cfg: DataConfig) -> None:
        _validate_config(cfg)
        self.cfg = cfg

    def __len__(self) -> int:
        return int(self.cfg.num_samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        image, depth = make_scene(self.cfg, int(idx))
        observed, targets = add_transparent_overlay(image, depth, cfg=self.cfg, idx=int(idx))
        return observed, targets


def get_dataloaders(cfg: DataConfig) -> tuple[DataLoader, DataLoader]:
    _validate_config(cfg)
    ds = SyntheticTransparentDepthDataset(cfg)
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
    "SyntheticTransparentDepthDataset",
    "add_transparent_overlay",
    "get_dataloaders",
    "make_scene",
]
