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
    palette_dim: int = 6


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
        radius = float(torch.empty((1,)).uniform_(0.2, 0.52, generator=generator).item())
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
        width = float(torch.empty((1,)).uniform_(0.1, 0.22, generator=generator).item())
        mask[0, torch.abs(yy - slope * xx - bias) <= width] = 1.0
    else:
        threshold = float(torch.empty((1,)).uniform_(-0.25, 0.25, generator=generator).item())
        mask[0, yy + 0.25 * xx >= threshold] = 1.0
    return mask


def _make_layout(image_size: int, generator: torch.Generator) -> torch.Tensor:
    primary = _shape_mask(image_size, generator)
    support = _shape_mask(image_size, generator)
    layout = 0.78 * primary + 0.24 * support
    layout += 0.04 * torch.rand(layout.shape, generator=generator, dtype=torch.float32)
    return layout.clamp(0.0, 1.0)


def _sample_palette_code(palette_dim: int, generator: torch.Generator) -> torch.Tensor:
    if palette_dim != 6:
        raise ValueError("palette_dim must be 6 for this toy lesson")
    foreground = torch.rand((3,), generator=generator, dtype=torch.float32)
    accent = torch.rand((3,), generator=generator, dtype=torch.float32)
    foreground = 0.25 + 0.7 * foreground
    accent = 0.05 + 0.75 * accent
    return torch.cat([foreground, accent], dim=0)


def _decode_palette_code(palette_code: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if int(palette_code.numel()) != 6:
        raise ValueError("palette_dim must be 6 for this toy lesson")
    foreground = palette_code[:3]
    accent = palette_code[3:]
    return foreground, accent


def _make_target(layout: torch.Tensor, palette_code: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    image_size = int(layout.shape[-1])
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, image_size, dtype=torch.float32),
        torch.linspace(-1.0, 1.0, image_size, dtype=torch.float32),
        indexing="ij",
    )
    structure = (layout > 0.3).to(dtype=torch.float32)
    contour = torch.clamp(layout - 0.58 * torch.roll(layout, shifts=1, dims=1), min=0.0, max=1.0)
    blend = (0.5 + 0.5 * xx).unsqueeze(0)
    waves = (0.5 + 0.5 * torch.sin(yy * 8.0 - xx * 4.5)).unsqueeze(0)

    foreground, accent = _decode_palette_code(palette_code.to(dtype=torch.float32))
    fg = foreground.view(3, 1, 1)
    accent = accent.view(3, 1, 1)
    neutral = 0.5 * (fg + accent)

    color_field = blend * fg + (1.0 - blend) * accent
    interior = structure * (0.35 + 0.65 * layout) * color_field
    highlights = contour * (0.25 * fg + 0.75 * neutral)
    background = (1.0 - structure) * (0.08 + 0.20 * waves) * (0.65 * accent + 0.35 * neutral)
    target = interior + highlights + background
    target += 0.02 * torch.randn((3, image_size, image_size), generator=generator, dtype=torch.float32)
    target += 0.01 * torch.rand((3, image_size, image_size), generator=generator, dtype=torch.float32)
    return target.clamp(0.0, 1.0)


def _make_toy_layout_palette_fusion_data(cfg: DataConfig) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(int(cfg.seed))
    layouts: list[torch.Tensor] = []
    palette_codes: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []

    for _ in range(int(cfg.num_samples)):
        layout = _make_layout(int(cfg.image_size), generator)
        palette_code = _sample_palette_code(int(cfg.palette_dim), generator)
        target = _make_target(layout, palette_code, generator)
        layouts.append(layout)
        palette_codes.append(palette_code)
        targets.append(target)

    return torch.stack(layouts, dim=0), torch.stack(palette_codes, dim=0), torch.stack(targets, dim=0)


class ToyLayoutPaletteFusionDataset(Dataset):
    def __init__(self, cfg: DataConfig) -> None:
        self.cfg = cfg
        self._layout, self._palette_code, self._target = _make_toy_layout_palette_fusion_data(cfg)

    def __len__(self) -> int:
        return int(self._target.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        i = int(idx)
        return self._layout[i], self._palette_code[i], self._target[i]


def get_dataloaders(cfg: DataConfig) -> tuple[DataLoader, DataLoader]:
    dataset: Dataset = ToyLayoutPaletteFusionDataset(cfg)
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


__all__ = ["DataConfig", "ToyLayoutPaletteFusionDataset", "get_dataloaders"]
