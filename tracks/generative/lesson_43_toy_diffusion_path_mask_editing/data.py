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


def _uniform(generator: torch.Generator, low: float, high: float) -> float:
    return float(torch.empty((), dtype=torch.float32).uniform_(low, high, generator=generator).item())


def _region_mask(image_size: int, generator: torch.Generator) -> torch.Tensor:
    yy, xx = torch.meshgrid(
        torch.arange(image_size, dtype=torch.float32),
        torch.arange(image_size, dtype=torch.float32),
        indexing="ij",
    )
    cx = _uniform(generator, 0.35 * image_size, 0.65 * image_size)
    cy = _uniform(generator, 0.35 * image_size, 0.65 * image_size)
    rx = _uniform(generator, 0.18 * image_size, 0.28 * image_size)
    ry = _uniform(generator, 0.14 * image_size, 0.24 * image_size)
    theta = _uniform(generator, -0.9, 0.9)
    cos_t = torch.cos(torch.tensor(theta, dtype=torch.float32))
    sin_t = torch.sin(torch.tensor(theta, dtype=torch.float32))
    x_shift = xx - cx
    y_shift = yy - cy
    xr = x_shift * cos_t + y_shift * sin_t
    yr = -x_shift * sin_t + y_shift * cos_t
    mask = ((xr / max(rx, 1e-6)) ** 2 + (yr / max(ry, 1e-6)) ** 2 <= 1.0).to(torch.float32)
    return mask.unsqueeze(0)


def _path_mask(region_mask: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    coords = torch.nonzero(region_mask[0] > 0.5, as_tuple=False)
    if coords.size(0) < 3:
        return region_mask.clone()

    path = torch.zeros_like(region_mask)
    size = int(region_mask.shape[-1])
    yy, xx = torch.meshgrid(
        torch.arange(size, dtype=torch.float32),
        torch.arange(size, dtype=torch.float32),
        indexing="ij",
    )
    num_waypoints = int(torch.randint(4, 7, (1,), generator=generator).item())
    waypoint_ids = torch.randperm(coords.size(0), generator=generator)[:num_waypoints]
    waypoints = coords[waypoint_ids].to(torch.float32)

    for segment_idx in range(num_waypoints - 1):
        y0, x0 = waypoints[segment_idx]
        y1, x1 = waypoints[segment_idx + 1]
        dy = y1 - y0
        dx = x1 - x0
        denom = torch.clamp(dx.pow(2) + dy.pow(2), min=1e-6)
        projection = ((xx - x0) * dx + (yy - y0) * dy) / denom
        projection = projection.clamp(0.0, 1.0)
        proj_x = x0 + projection * dx
        proj_y = y0 + projection * dy
        radius = _uniform(generator, 0.9, 1.8)
        patch = (((xx - proj_x) ** 2 + (yy - proj_y) ** 2) <= radius**2).to(torch.float32)
        path[0] = torch.maximum(path[0], patch)

    return (path * region_mask).clamp(0.0, 1.0)


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


def _make_source(region_mask: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    base = _texture(region_mask.shape[-1], generator)
    background = 0.05 + 0.07 * torch.rand(region_mask.shape, generator=generator, dtype=torch.float32)
    source = background * (1.0 - region_mask) + (0.18 + 0.72 * base) * region_mask
    source += 0.02 * torch.rand(region_mask.shape, generator=generator, dtype=torch.float32)
    return source.clamp(0.0, 1.0)


def _make_target(*, source: torch.Tensor, path_mask: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    edited = 0.60 * torch.flip(source, dims=(2,)) + 0.40 * torch.roll(source, shifts=3, dims=1)
    edited = 0.08 + 0.92 * edited
    target = source * (1.0 - path_mask) + edited * path_mask
    target += 0.02 * torch.randn(source.shape, generator=generator, dtype=torch.float32)
    target += 0.01 * torch.rand(source.shape, generator=generator, dtype=torch.float32)
    return target.clamp(0.0, 1.0)


def _make_toy_path_mask_editing_data(cfg: DataConfig) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(int(cfg.seed))
    sources: list[torch.Tensor] = []
    path_masks: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []

    for _ in range(int(cfg.num_samples)):
        region_mask = _region_mask(int(cfg.image_size), generator)
        path_mask = _path_mask(region_mask, generator)
        source = _make_source(region_mask, generator)
        target = _make_target(source=source, path_mask=path_mask, generator=generator)
        sources.append(source)
        path_masks.append(path_mask)
        targets.append(target)

    return torch.stack(sources, dim=0), torch.stack(path_masks, dim=0), torch.stack(targets, dim=0)


class ToyPathMaskEditingDataset(Dataset):
    def __init__(self, cfg: DataConfig) -> None:
        self.cfg = cfg
        self._source, self._path_mask, self._target = _make_toy_path_mask_editing_data(cfg)

    def __len__(self) -> int:
        return int(self._target.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        i = int(idx)
        return self._source[i], self._path_mask[i], self._target[i]


def get_dataloaders(cfg: DataConfig) -> tuple[DataLoader, DataLoader]:
    dataset: Dataset = ToyPathMaskEditingDataset(cfg)
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


__all__ = ["DataConfig", "ToyPathMaskEditingDataset", "get_dataloaders"]
