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
    upscale_factor: int = 2
    blur_kernel_size: int = 3
    noise_std: float = 0.01
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
    if int(cfg.in_channels) <= 0:
        raise ValueError("in_channels must be > 0")
    if int(cfg.upscale_factor) != 2:
        raise ValueError("Only upscale_factor=2 is supported in v1")
    if int(cfg.image_size) % int(cfg.upscale_factor) != 0:
        raise ValueError("image_size must be divisible by upscale_factor")
    if int(cfg.blur_kernel_size) <= 0 or int(cfg.blur_kernel_size) % 2 == 0:
        raise ValueError("blur_kernel_size must be a positive odd integer")
    if float(cfg.noise_std) < 0.0:
        raise ValueError("noise_std must be >= 0")
    if int(cfg.min_shapes) <= 0 or int(cfg.max_shapes) < int(cfg.min_shapes):
        raise ValueError("invalid shape count range")


def _sample_color(generator: torch.Generator, in_channels: int) -> torch.Tensor:
    return torch.rand((int(in_channels),), generator=generator, dtype=torch.float32) * 0.85 + 0.15


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


def make_high_res_image(cfg: DataConfig, idx: int) -> torch.Tensor:
    _validate_config(cfg)
    h = int(cfg.image_size)
    c = int(cfg.in_channels)
    seed = int(cfg.seed) * 1_000_003 + int(idx) * 97
    g = torch.Generator().manual_seed(seed)

    image = torch.zeros((c, h, h), dtype=torch.float32)

    yy = torch.linspace(0.0, 1.0, h, dtype=torch.float32).view(1, h, 1)
    xx = torch.linspace(0.0, 1.0, h, dtype=torch.float32).view(1, 1, h)
    image = image + 0.08 * yy + 0.05 * xx
    image = image.expand(c, h, h).clone()

    num_shapes = int(
        torch.randint(
            low=int(cfg.min_shapes),
            high=int(cfg.max_shapes) + 1,
            size=(1,),
            generator=g,
        ).item()
    )
    min_size = max(4, h // 8)
    max_size = max(min_size + 1, h // 2)

    for shape_idx in range(num_shapes):
        color = _sample_color(g, c)
        if shape_idx % 2 == 0:
            rect_h = int(torch.randint(min_size, max_size + 1, (1,), generator=g).item())
            rect_w = int(torch.randint(min_size, max_size + 1, (1,), generator=g).item())
            top = int(torch.randint(0, h - rect_h + 1, (1,), generator=g).item())
            left = int(torch.randint(0, h - rect_w + 1, (1,), generator=g).item())
            _draw_rectangle(image, top=top, left=left, height=rect_h, width=rect_w, color=color)
        else:
            radius = int(torch.randint(min_size // 2, max_size // 2 + 1, (1,), generator=g).item())
            cy = int(torch.randint(radius, h - radius + 1, (1,), generator=g).item())
            cx = int(torch.randint(radius, h - radius + 1, (1,), generator=g).item())
            _draw_circle(image, cy=cy, cx=cx, radius=radius, color=color)

    return image.clamp(0.0, 1.0)


def degrade_to_low_res(hr: torch.Tensor, *, cfg: DataConfig, idx: int) -> torch.Tensor:
    if hr.ndim != 3:
        raise ValueError(f"Expected HR image shape (C, H, W), got {tuple(hr.shape)}")
    _validate_config(cfg)

    x = hr.unsqueeze(0)
    kernel = int(cfg.blur_kernel_size)
    pad = kernel // 2
    if kernel > 1:
        weight = torch.ones((int(cfg.in_channels), 1, kernel, kernel), dtype=x.dtype) / float(
            kernel * kernel
        )
        x = F.conv2d(x, weight, padding=pad, groups=int(cfg.in_channels))

    out_h = int(cfg.image_size) // int(cfg.upscale_factor)
    x = F.interpolate(x, size=(out_h, out_h), mode="bicubic", align_corners=False)

    noise_std = float(cfg.noise_std)
    if noise_std > 0.0:
        g = torch.Generator().manual_seed(int(cfg.seed) * 1_000_003 + int(idx) * 97 + 13)
        noise = torch.randn(x.shape, generator=g, dtype=x.dtype)
        x = x + noise * noise_std
    return x.squeeze(0).clamp(0.0, 1.0)


class ToySuperResolutionDataset(Dataset):
    def __init__(self, cfg: DataConfig) -> None:
        _validate_config(cfg)
        self.cfg = cfg

    def __len__(self) -> int:
        return int(self.cfg.num_samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        hr = make_high_res_image(self.cfg, int(idx))
        lr = degrade_to_low_res(hr, cfg=self.cfg, idx=int(idx))
        return lr, hr


def get_dataloaders(cfg: DataConfig) -> tuple[DataLoader, DataLoader]:
    _validate_config(cfg)
    ds = ToySuperResolutionDataset(cfg)
    train_idx, val_idx = train_val_split_indices(
        n=len(ds),
        val_fraction=float(cfg.val_fraction),
        seed=int(cfg.seed),
    )
    train_ds = Subset(ds, train_idx)
    val_ds = Subset(ds, val_idx)

    loader_kwargs = {
        "batch_size": int(cfg.batch_size),
        "num_workers": int(cfg.num_workers),
        "pin_memory": False,
    }
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    return train_loader, val_loader


__all__ = [
    "DataConfig",
    "ToySuperResolutionDataset",
    "degrade_to_low_res",
    "get_dataloaders",
    "make_high_res_image",
]
