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
    lighting_dim: int = 4


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
    layout = 0.76 * primary + 0.28 * support
    layout += 0.05 * torch.rand(layout.shape, generator=generator, dtype=torch.float32)
    return layout.clamp(0.0, 1.0)


def _sample_lighting_code(lighting_dim: int, generator: torch.Generator) -> torch.Tensor:
    if lighting_dim != 4:
        raise ValueError("lighting_dim must be 4 for this toy lesson")
    light_x = float(torch.empty((1,)).uniform_(-1.0, 1.0, generator=generator).item())
    light_y = float(torch.empty((1,)).uniform_(-1.0, 1.0, generator=generator).item())
    ambient = float(torch.empty((1,)).uniform_(0.18, 0.42, generator=generator).item())
    warmth = float(torch.empty((1,)).uniform_(0.0, 1.0, generator=generator).item())
    return torch.tensor([light_x, light_y, ambient, warmth], dtype=torch.float32)


def _decode_lighting_code(lighting_code: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if int(lighting_code.numel()) != 4:
        raise ValueError("lighting_dim must be 4 for this toy lesson")
    direction = lighting_code[:2].to(dtype=torch.float32)
    direction = direction / direction.norm().clamp(min=1e-6)
    ambient = lighting_code[2:3].to(dtype=torch.float32)
    warmth = lighting_code[3:4].to(dtype=torch.float32)
    return direction, ambient, warmth


def _make_target(layout: torch.Tensor, lighting_code: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    image_size = int(layout.shape[-1])
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, image_size, dtype=torch.float32),
        torch.linspace(-1.0, 1.0, image_size, dtype=torch.float32),
        indexing="ij",
    )
    direction, ambient, warmth = _decode_lighting_code(lighting_code)
    structure = (layout > 0.3).to(dtype=torch.float32)
    contour = torch.clamp(layout - 0.55 * torch.roll(layout, shifts=1, dims=2), min=0.0, max=1.0)
    directional = (0.5 + 0.5 * (direction[0] * xx + direction[1] * yy)).unsqueeze(0)
    soft_fill = (0.5 + 0.5 * torch.cos(xx * 3.5 - yy * 2.0)).unsqueeze(0)

    warm_rgb = torch.tensor([1.00, 0.78, 0.50], dtype=torch.float32).view(3, 1, 1)
    cool_rgb = torch.tensor([0.42, 0.62, 1.00], dtype=torch.float32).view(3, 1, 1)
    neutral_rgb = torch.tensor([0.42, 0.44, 0.50], dtype=torch.float32).view(3, 1, 1)
    key_color = warmth.view(1, 1, 1) * warm_rgb + (1.0 - warmth.view(1, 1, 1)) * cool_rgb
    rim_color = 0.55 * key_color + 0.45 * torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32).view(3, 1, 1)
    background_color = 0.55 * neutral_rgb + 0.45 * cool_rgb

    shading = (ambient.view(1, 1, 1) + (1.0 - ambient.view(1, 1, 1)) * directional).clamp(0.0, 1.0)
    interior = structure * (0.22 + 0.78 * layout) * shading * key_color
    highlights = contour * (0.35 + 0.65 * soft_fill) * rim_color
    background = (1.0 - structure) * (0.05 + 0.18 * soft_fill) * background_color
    target = interior + highlights + background
    target += 0.02 * torch.randn((3, image_size, image_size), generator=generator, dtype=torch.float32)
    target += 0.01 * torch.rand((3, image_size, image_size), generator=generator, dtype=torch.float32)
    return target.clamp(0.0, 1.0)


def _make_toy_layout_lighting_fusion_data(cfg: DataConfig) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(int(cfg.seed))
    layouts: list[torch.Tensor] = []
    lighting_codes: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []

    for _ in range(int(cfg.num_samples)):
        layout = _make_layout(int(cfg.image_size), generator)
        lighting_code = _sample_lighting_code(int(cfg.lighting_dim), generator)
        target = _make_target(layout, lighting_code, generator)
        layouts.append(layout)
        lighting_codes.append(lighting_code)
        targets.append(target)

    return torch.stack(layouts, dim=0), torch.stack(lighting_codes, dim=0), torch.stack(targets, dim=0)


class ToyLayoutLightingFusionDataset(Dataset):
    def __init__(self, cfg: DataConfig) -> None:
        self.cfg = cfg
        self._layout, self._lighting_code, self._target = _make_toy_layout_lighting_fusion_data(cfg)

    def __len__(self) -> int:
        return int(self._target.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        i = int(idx)
        return self._layout[i], self._lighting_code[i], self._target[i]


def get_dataloaders(cfg: DataConfig) -> tuple[DataLoader, DataLoader]:
    dataset: Dataset = ToyLayoutLightingFusionDataset(cfg)
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


__all__ = ["DataConfig", "ToyLayoutLightingFusionDataset", "get_dataloaders"]
