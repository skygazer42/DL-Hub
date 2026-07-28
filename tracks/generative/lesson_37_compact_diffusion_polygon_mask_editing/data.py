from __future__ import annotations

from dataclasses import dataclass

import numpy as np
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


def _uniform(generator: torch.Generator, low: float, high: float) -> float:
    return float(torch.empty((), dtype=torch.float32).uniform_(low, high, generator=generator).item())


def _polygon_mask(image_size: int, generator: torch.Generator) -> torch.Tensor:
    num_vertices = int(torch.randint(3, 7, (1,), generator=generator).item())
    center_x = _uniform(generator, 0.35 * image_size, 0.65 * image_size)
    center_y = _uniform(generator, 0.35 * image_size, 0.65 * image_size)
    radius_low = 0.14 * image_size
    radius_high = 0.28 * image_size

    angles = torch.sort(torch.rand((num_vertices,), generator=generator) * (2.0 * torch.pi)).values.numpy()
    radii = (
        radius_low
        + (radius_high - radius_low) * torch.rand((num_vertices,), generator=generator).numpy()
    )
    xs = np.clip(center_x + radii * np.cos(angles), 1.0, float(image_size - 2))
    ys = np.clip(center_y + radii * np.sin(angles), 1.0, float(image_size - 2))
    vertices = np.stack([xs, ys], axis=1).astype(np.float32)

    yy, xx = np.mgrid[0:image_size, 0:image_size].astype(np.float32)
    inside = np.zeros((image_size, image_size), dtype=bool)
    j = num_vertices - 1
    for i in range(num_vertices):
        xi, yi = vertices[i]
        xj, yj = vertices[j]
        cross = ((yi > yy) != (yj > yy)) & (
            xx < (xj - xi) * (yy - yi) / (max(abs(yj - yi), 1e-6) * np.sign(yj - yi or 1.0)) + xi
        )
        inside ^= cross
        j = i
    return torch.from_numpy(inside.astype(np.float32)).unsqueeze(0)


def _texture(image_size: int, generator: torch.Generator) -> torch.Tensor:
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, image_size, dtype=torch.float32),
        torch.linspace(-1.0, 1.0, image_size, dtype=torch.float32),
        indexing="ij",
    )
    mode = int(torch.randint(0, 4, (1,), generator=generator).item())
    if mode == 0:
        phase = float(torch.empty((1,)).uniform_(-3.14, 3.14, generator=generator).item())
        image = 0.5 + 0.5 * torch.sin(xx * 7.5 + phase)
    elif mode == 1:
        image = 0.5 + 0.5 * torch.cos(yy * 9.0)
    elif mode == 2:
        image = (((xx * 6.0).floor() + (yy * 6.0).floor()) % 2.0 == 0.0).to(torch.float32)
    else:
        image = 0.5 + 0.5 * torch.cos((xx.pow(2) + yy.pow(2)).sqrt() * 10.0)
    image = image.unsqueeze(0)
    image += 0.05 * torch.rand((1, image_size, image_size), generator=generator, dtype=torch.float32)
    return image.clamp(0.0, 1.0)


def _make_source(polygon_mask: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    base = _texture(polygon_mask.shape[-1], generator)
    background = 0.05 + 0.07 * torch.rand(polygon_mask.shape, generator=generator, dtype=torch.float32)
    source = background * (1.0 - polygon_mask) + (0.18 + 0.72 * base) * polygon_mask
    source += 0.02 * torch.rand(polygon_mask.shape, generator=generator, dtype=torch.float32)
    return source.clamp(0.0, 1.0)


def _make_target(
    *,
    source: torch.Tensor,
    polygon_mask: torch.Tensor,
    generator: torch.Generator,
) -> torch.Tensor:
    edited = 0.45 * torch.flip(source, dims=(2,)) + 0.55 * torch.roll(source, shifts=2, dims=1)
    edited = 0.10 + 0.90 * edited
    target = source * (1.0 - polygon_mask) + edited * polygon_mask
    target += 0.02 * torch.randn(source.shape, generator=generator, dtype=torch.float32)
    target += 0.01 * torch.rand(source.shape, generator=generator, dtype=torch.float32)
    return target.clamp(0.0, 1.0)


def _make_synthetic_polygon_mask_editing_data(cfg: DataConfig) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(int(cfg.seed))
    sources: list[torch.Tensor] = []
    polygon_masks: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []

    for _ in range(int(cfg.num_samples)):
        polygon_mask = _polygon_mask(int(cfg.image_size), generator)
        source = _make_source(polygon_mask, generator)
        target = _make_target(source=source, polygon_mask=polygon_mask, generator=generator)
        sources.append(source)
        polygon_masks.append(polygon_mask)
        targets.append(target)

    return (
        torch.stack(sources, dim=0),
        torch.stack(polygon_masks, dim=0),
        torch.stack(targets, dim=0),
    )


class SyntheticPolygonMaskEditingDataset(Dataset):
    def __init__(self, cfg: DataConfig) -> None:
        self.cfg = cfg
        self._source, self._polygon_mask, self._target = _make_synthetic_polygon_mask_editing_data(cfg)

    def __len__(self) -> int:
        return int(self._target.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        i = int(idx)
        return self._source[i], self._polygon_mask[i], self._target[i]


def get_dataloaders(cfg: DataConfig) -> tuple[DataLoader, DataLoader]:
    dataset: Dataset = SyntheticPolygonMaskEditingDataset(cfg)
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


__all__ = ["DataConfig", "SyntheticPolygonMaskEditingDataset", "get_dataloaders"]
