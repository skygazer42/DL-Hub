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
    shadow_strength_min: float = 0.25
    shadow_strength_max: float = 0.55
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
    if float(cfg.shadow_strength_min) < 0.0 or float(cfg.shadow_strength_max) > 1.0:
        raise ValueError("shadow strengths must be in [0, 1]")
    if float(cfg.shadow_strength_max) < float(cfg.shadow_strength_min):
        raise ValueError("shadow_strength_max must be >= shadow_strength_min")
    if int(cfg.min_shapes) <= 0 or int(cfg.max_shapes) < int(cfg.min_shapes):
        raise ValueError("invalid shape count range")


def _sample_color(generator: torch.Generator) -> torch.Tensor:
    return torch.rand((3,), generator=generator, dtype=torch.float32) * 0.7 + 0.2


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


def _draw_ellipse(
    image: torch.Tensor,
    *,
    cy: int,
    cx: int,
    ry: int,
    rx: int,
    color: torch.Tensor,
) -> None:
    h, w = image.shape[-2:]
    yy = torch.arange(h, dtype=torch.float32).view(h, 1)
    xx = torch.arange(w, dtype=torch.float32).view(1, w)
    norm = ((yy - float(cy)) / float(max(1, ry))).pow(2) + ((xx - float(cx)) / float(max(1, rx))).pow(2)
    mask = norm <= 1.0
    image[:, mask] = color.view(-1, 1)


def _render_clean_image(cfg: DataConfig, idx: int) -> torch.Tensor:
    _validate_config(cfg)
    size = int(cfg.image_size)
    gen = torch.Generator().manual_seed(int(cfg.seed) * 1_000_003 + int(idx) * 71)

    yy = torch.linspace(0.0, 1.0, size, dtype=torch.float32).view(size, 1)
    xx = torch.linspace(0.0, 1.0, size, dtype=torch.float32).view(1, size)
    image = torch.zeros((3, size, size), dtype=torch.float32)
    image[0] = 0.10 + 0.35 * yy + 0.12 * xx
    image[1] = 0.12 + 0.30 * yy + 0.22 * xx
    image[2] = 0.16 + 0.18 * yy + 0.34 * xx

    min_size = max(4, size // 8)
    max_size = max(min_size + 1, size // 2)
    num_shapes = int(
        torch.randint(
            low=int(cfg.min_shapes),
            high=int(cfg.max_shapes) + 1,
            size=(1,),
            generator=gen,
        ).item()
    )
    for shape_idx in range(num_shapes):
        color = _sample_color(gen)
        if shape_idx % 2 == 0:
            rect_h = int(torch.randint(min_size, max_size + 1, (1,), generator=gen).item())
            rect_w = int(torch.randint(min_size, max_size + 1, (1,), generator=gen).item())
            top = int(torch.randint(0, size - rect_h + 1, (1,), generator=gen).item())
            left = int(torch.randint(0, size - rect_w + 1, (1,), generator=gen).item())
            _draw_rectangle(image, top=top, left=left, height=rect_h, width=rect_w, color=color)
        else:
            ry = int(torch.randint(max(2, min_size // 2), max_size // 2 + 1, (1,), generator=gen).item())
            rx = int(torch.randint(max(2, min_size // 2), max_size // 2 + 1, (1,), generator=gen).item())
            cy = int(torch.randint(ry, size - ry + 1, (1,), generator=gen).item())
            cx = int(torch.randint(rx, size - rx + 1, (1,), generator=gen).item())
            _draw_ellipse(image, cy=cy, cx=cx, ry=ry, rx=rx, color=color)
    return image.clamp(0.0, 1.0)


def _render_shadow_mask(cfg: DataConfig, idx: int) -> torch.Tensor:
    _validate_config(cfg)
    size = int(cfg.image_size)
    gen = torch.Generator().manual_seed(int(cfg.seed) * 1_000_003 + int(idx) * 71 + 29)
    mask = torch.zeros((1, size, size), dtype=torch.float32)

    min_size = max(4, size // 8)
    max_size = max(min_size + 1, size // 2)
    num_shapes = int(torch.randint(1, 4, (1,), generator=gen).item())
    for shape_idx in range(num_shapes):
        if shape_idx % 2 == 0:
            rect_h = int(torch.randint(min_size, max_size + 1, (1,), generator=gen).item())
            rect_w = int(torch.randint(min_size, max_size + 1, (1,), generator=gen).item())
            top = int(torch.randint(0, size - rect_h + 1, (1,), generator=gen).item())
            left = int(torch.randint(0, size - rect_w + 1, (1,), generator=gen).item())
            mask[:, top : top + rect_h, left : left + rect_w] = 1.0
        else:
            ry = int(torch.randint(max(2, min_size // 2), max_size // 2 + 1, (1,), generator=gen).item())
            rx = int(torch.randint(max(2, min_size // 2), max_size // 2 + 1, (1,), generator=gen).item())
            cy = int(torch.randint(ry, size - ry + 1, (1,), generator=gen).item())
            cx = int(torch.randint(rx, size - rx + 1, (1,), generator=gen).item())
            yy = torch.arange(size, dtype=torch.float32).view(size, 1)
            xx = torch.arange(size, dtype=torch.float32).view(1, size)
            ellipse = (((yy - float(cy)) / float(max(1, ry))).pow(2) + ((xx - float(cx)) / float(max(1, rx))).pow(2)) <= 1.0
            mask[:, ellipse] = 1.0

    mask = torch.nn.functional.avg_pool2d(mask.unsqueeze(0), kernel_size=3, stride=1, padding=1).squeeze(0)
    return mask.clamp(0.0, 1.0)


def _compute_boundary(mask: torch.Tensor) -> torch.Tensor:
    dx = torch.abs(mask - torch.roll(mask, shifts=1, dims=2))
    dy = torch.abs(mask - torch.roll(mask, shifts=1, dims=1))
    boundary = torch.clamp(dx + dy, 0.0, 1.0)
    return boundary


def synthesize_shadowed_image(
    clean: torch.Tensor,
    shadow_mask: torch.Tensor,
    *,
    cfg: DataConfig,
    idx: int,
) -> torch.Tensor:
    _validate_config(cfg)
    if clean.ndim != 3 or shadow_mask.ndim != 3:
        raise ValueError("Expected clean and shadow_mask to have shape (C,H,W) and (1,H,W)")
    gen = torch.Generator().manual_seed(int(cfg.seed) * 1_000_003 + int(idx) * 71 + 53)
    strength = float(cfg.shadow_strength_min) + float(torch.rand((), generator=gen).item()) * float(
        cfg.shadow_strength_max - cfg.shadow_strength_min
    )
    attenuation = 1.0 - float(strength) * shadow_mask
    tint = 0.04 * shadow_mask.repeat(clean.shape[0], 1, 1)
    shadowed = clean * attenuation - tint
    noise = 0.01 * torch.rand(clean.shape, generator=gen, dtype=torch.float32)
    return (shadowed + noise).clamp(0.0, 1.0)


class SyntheticShadowDetectionDataset(Dataset):
    def __init__(self, cfg: DataConfig) -> None:
        _validate_config(cfg)
        self.cfg = cfg

    def __len__(self) -> int:
        return int(self.cfg.num_samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        clean = _render_clean_image(self.cfg, int(idx))
        shadow_mask = _render_shadow_mask(self.cfg, int(idx))
        boundary = _compute_boundary(shadow_mask)
        shadowed = synthesize_shadowed_image(clean, shadow_mask, cfg=self.cfg, idx=int(idx))
        return shadowed, {
            "shadow_mask": shadow_mask,
            "boundary": boundary,
            "lit_image": clean,
        }


def get_dataloaders(cfg: DataConfig) -> tuple[DataLoader, DataLoader]:
    _validate_config(cfg)
    ds = SyntheticShadowDetectionDataset(cfg)
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
    "SyntheticShadowDetectionDataset",
    "get_dataloaders",
    "synthesize_shadowed_image",
]
