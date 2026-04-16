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
    blur_kernel_size: int = 7
    blur_sigma: float = 1.2


def _validate_config(cfg: DataConfig) -> None:
    if int(cfg.num_samples) <= 0:
        raise ValueError("num_samples must be > 0")
    if int(cfg.batch_size) <= 0:
        raise ValueError("batch_size must be > 0")
    if int(cfg.image_size) < 16:
        raise ValueError("image_size must be >= 16")
    if not (0.0 < float(cfg.val_fraction) < 1.0):
        raise ValueError("val_fraction must be in (0,1)")
    if int(cfg.in_channels) != 3:
        raise ValueError("in_channels must be 3 for v1")
    if int(cfg.min_shapes) <= 0 or int(cfg.max_shapes) < int(cfg.min_shapes):
        raise ValueError("invalid shape range")
    if int(cfg.blur_kernel_size) < 3 or int(cfg.blur_kernel_size) % 2 == 0:
        raise ValueError("blur_kernel_size must be an odd integer >= 3")
    if float(cfg.blur_sigma) <= 0.0:
        raise ValueError("blur_sigma must be > 0")


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
    image[:, top : top + height, left : left + width] = color.view(3, 1, 1)
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
    image[:, mask] = color.view(3, 1)
    depth[:, mask] = float(depth_value)


def make_clean_image_and_depth(cfg: DataConfig, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
    _validate_config(cfg)
    size = int(cfg.image_size)
    seed = int(cfg.seed) * 1_000_003 + int(idx) * 97
    gen = torch.Generator().manual_seed(seed)

    yy = torch.linspace(0.0, 1.0, size, dtype=torch.float32).view(1, size, 1)
    xx = torch.linspace(0.0, 1.0, size, dtype=torch.float32).view(1, 1, size)
    clean = torch.zeros((3, size, size), dtype=torch.float32)
    clean[0] = 0.10 + 0.30 * yy.squeeze(0) + 0.08 * xx.squeeze(0)
    clean[1] = 0.12 + 0.20 * yy.squeeze(0)
    clean[2] = 0.14 + 0.24 * xx.squeeze(0)

    depth = (0.20 + 0.80 * yy).expand(1, size, size).clone()

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
        depth_value = float(torch.rand((), generator=gen).item() * 0.70 + 0.05)
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


def _gaussian_kernel(ksize: int, sigma: float, dtype: torch.dtype) -> torch.Tensor:
    coords = torch.arange(ksize, dtype=dtype) - (ksize - 1) / 2.0
    gauss = torch.exp(-(coords**2) / (2.0 * (sigma**2)))
    gauss = gauss / gauss.sum()
    kernel_2d = gauss[:, None] @ gauss[None, :]
    return kernel_2d / kernel_2d.sum()


def _blur_image(image: torch.Tensor, ksize: int, sigma: float) -> torch.Tensor:
    if image.ndim != 3:
        raise ValueError("image must have shape (C,H,W)")
    c = int(image.shape[0])
    kernel = _gaussian_kernel(ksize, sigma, image.dtype).view(1, 1, ksize, ksize)
    weight = kernel.repeat(c, 1, 1, 1)
    padded = torch.nn.functional.pad(
        image.unsqueeze(0),
        (ksize // 2, ksize // 2, ksize // 2, ksize // 2),
        mode="reflect",
    )
    return torch.nn.functional.conv2d(padded, weight, groups=c).squeeze(0)


def synthesize_focus_pair(
    clean: torch.Tensor,
    depth: torch.Tensor,
    *,
    cfg: DataConfig,
    idx: int,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    if clean.ndim != 3 or depth.ndim != 3:
        raise ValueError("Expected clean and depth tensors with shape (C,H,W) and (1,H,W)")
    _validate_config(cfg)
    gen = torch.Generator().manual_seed(int(cfg.seed) * 1_000_003 + int(idx) * 97 + 17)
    split = float(torch.rand((), generator=gen).item()) * 0.3 + 0.35
    near_mask = (depth <= split).to(clean.dtype)
    far_mask = 1.0 - near_mask

    blurred = _blur_image(clean, int(cfg.blur_kernel_size), float(cfg.blur_sigma))
    near_focus = clean * near_mask + blurred * far_mask
    far_focus = clean * far_mask + blurred * near_mask
    return {"near_focus": near_focus.clamp(0.0, 1.0), "far_focus": far_focus.clamp(0.0, 1.0)}, clean


class SyntheticImageFusionDataset(Dataset):
    def __init__(self, cfg: DataConfig) -> None:
        _validate_config(cfg)
        self.cfg = cfg

    def __len__(self) -> int:
        return int(self.cfg.num_samples)

    def __getitem__(self, idx: int) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        clean, depth = make_clean_image_and_depth(self.cfg, int(idx))
        return synthesize_focus_pair(clean, depth, cfg=self.cfg, idx=int(idx))


def get_dataloaders(cfg: DataConfig) -> tuple[DataLoader, DataLoader]:
    _validate_config(cfg)
    ds = SyntheticImageFusionDataset(cfg)
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
    "SyntheticImageFusionDataset",
    "get_dataloaders",
    "make_clean_image_and_depth",
    "synthesize_focus_pair",
]
