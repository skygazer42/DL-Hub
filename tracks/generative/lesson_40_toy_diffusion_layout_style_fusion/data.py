from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from dlhub.data.splits import train_val_split_indices


@dataclass(frozen=True)
class DataConfig:
    batch_size: int = 32
    image_size: int = 28
    num_workers: int = 0
    num_samples: int = 256
    seed: int = 0
    val_fraction: float = 0.2
    style_dim: int = 3


def _shape_mask(image_size: int, generator: torch.Generator) -> torch.Tensor:
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, image_size, dtype=torch.float32),
        torch.linspace(-1.0, 1.0, image_size, dtype=torch.float32),
        indexing="ij",
    )
    mask = torch.zeros((1, image_size, image_size), dtype=torch.float32)
    mode = int(torch.randint(0, 5, (1,), generator=generator).item())
    if mode == 0:
        cx = float(torch.empty((1,)).uniform_(-0.45, 0.45, generator=generator).item())
        cy = float(torch.empty((1,)).uniform_(-0.45, 0.45, generator=generator).item())
        radius = float(torch.empty((1,)).uniform_(0.2, 0.5, generator=generator).item())
        mask[0, (xx - cx).pow(2) + (yy - cy).pow(2) <= radius**2] = 1.0
    elif mode == 1:
        y1 = int(torch.randint(2, image_size // 2, (1,), generator=generator).item())
        y2 = int(torch.randint(image_size // 2, image_size - 2, (1,), generator=generator).item())
        x1 = int(torch.randint(2, image_size // 2, (1,), generator=generator).item())
        x2 = int(torch.randint(image_size // 2, image_size - 2, (1,), generator=generator).item())
        mask[:, y1:y2, x1:x2] = 1.0
    elif mode == 2:
        center = int(torch.randint(5, image_size - 5, (1,), generator=generator).item())
        thickness = int(torch.randint(1, 3, (1,), generator=generator).item())
        mask[:, center - thickness : center + thickness + 1, :] = 1.0
        mask[:, :, center - thickness : center + thickness + 1] = 1.0
    elif mode == 3:
        slope = float(torch.empty((1,)).uniform_(-0.9, 0.9, generator=generator).item())
        bias = float(torch.empty((1,)).uniform_(-0.35, 0.35, generator=generator).item())
        width = float(torch.empty((1,)).uniform_(0.1, 0.24, generator=generator).item())
        mask[0, torch.abs(yy - slope * xx - bias) <= width] = 1.0
    else:
        threshold = float(torch.empty((1,)).uniform_(-0.25, 0.25, generator=generator).item())
        mask[0, yy - 0.35 * xx >= threshold] = 1.0
    return mask


def _make_layout(image_size: int, generator: torch.Generator) -> torch.Tensor:
    base = _shape_mask(image_size, generator)
    support = _shape_mask(image_size, generator)
    layout = 0.72 * base + 0.28 * support
    layout += 0.04 * torch.rand(layout.shape, generator=generator, dtype=torch.float32)
    return layout.clamp(0.0, 1.0)


def _sample_style_code(style_dim: int, generator: torch.Generator) -> torch.Tensor:
    raw = torch.rand((style_dim,), generator=generator, dtype=torch.float32)
    raw = raw + 0.15
    return (raw / raw.sum()).to(torch.float32)


def _style_bases(image_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, image_size, dtype=torch.float32),
        torch.linspace(-1.0, 1.0, image_size, dtype=torch.float32),
        indexing="ij",
    )
    stripes = 0.5 + 0.5 * torch.sin(xx * 8.5)
    rings = 0.5 + 0.5 * torch.cos((xx.pow(2) + yy.pow(2)).sqrt() * 12.0)
    checker = (((xx * 6.0).floor() + (yy * 6.0).floor()) % 2.0 == 0.0).to(torch.float32)
    return stripes, rings, checker


def _make_style_map(style_code: torch.Tensor, image_size: int, generator: torch.Generator) -> torch.Tensor:
    if int(style_code.numel()) != 3:
        raise ValueError("style_dim must be 3 for this toy lesson")
    bases = _style_bases(image_size)
    style = float(style_code[0]) * bases[0] + float(style_code[1]) * bases[1] + float(style_code[2]) * bases[2]
    style = style.unsqueeze(0)
    style += 0.04 * torch.rand((1, image_size, image_size), generator=generator, dtype=torch.float32)
    return style.clamp(0.0, 1.0)


def _make_target(layout: torch.Tensor, style_code: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    image_size = int(layout.shape[-1])
    structure = (layout > 0.3).to(layout.dtype)
    style = _make_style_map(style_code, image_size, generator)
    contour = 0.25 * torch.roll(layout, shifts=1, dims=1) + 0.75 * layout
    foreground = (0.18 + 0.72 * style) * structure
    background = (0.05 + 0.16 * (1.0 - layout)) * (1.0 - structure)
    target = foreground + background
    target = 0.70 * target + 0.30 * contour
    target += 0.02 * torch.randn(layout.shape, generator=generator, dtype=torch.float32)
    target += 0.01 * torch.rand(layout.shape, generator=generator, dtype=torch.float32)
    return target.clamp(0.0, 1.0)


def _make_toy_layout_style_fusion_data(cfg: DataConfig) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(int(cfg.seed))
    layouts: list[torch.Tensor] = []
    style_codes: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []

    style_dim = int(cfg.style_dim)
    if style_dim != 3:
        raise ValueError("style_dim must be 3 for this toy lesson")

    for _ in range(int(cfg.num_samples)):
        layout = _make_layout(int(cfg.image_size), generator)
        style_code = _sample_style_code(style_dim, generator)
        target = _make_target(layout, style_code, generator)
        layouts.append(layout)
        style_codes.append(style_code)
        targets.append(target)

    return torch.stack(layouts, dim=0), torch.stack(style_codes, dim=0), torch.stack(targets, dim=0)


class ToyLayoutStyleFusionDataset(Dataset):
    def __init__(self, cfg: DataConfig) -> None:
        self.cfg = cfg
        self._layout, self._style_code, self._target = _make_toy_layout_style_fusion_data(cfg)

    def __len__(self) -> int:
        return int(self._target.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        i = int(idx)
        return self._layout[i], self._style_code[i], self._target[i]


def get_dataloaders(cfg: DataConfig) -> tuple[DataLoader, DataLoader]:
    dataset: Dataset = ToyLayoutStyleFusionDataset(cfg)
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


__all__ = ["DataConfig", "ToyLayoutStyleFusionDataset", "get_dataloaders"]
