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


def _scribble_mask(region_mask: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    coords = torch.nonzero(region_mask[0] > 0.5, as_tuple=False)
    if coords.numel() == 0:
        return region_mask.clone()

    scribble = torch.zeros_like(region_mask)
    num_points = int(min(max(6, coords.size(0) // 48), 24))
    choice = torch.randperm(coords.size(0), generator=generator)[:num_points]
    size = region_mask.shape[-1]
    yy, xx = torch.meshgrid(
        torch.arange(size, dtype=torch.float32),
        torch.arange(size, dtype=torch.float32),
        indexing="ij",
    )
    for index in choice.tolist():
        cy, cx = coords[index].tolist()
        radius = _uniform(generator, 1.1, 2.4)
        patch = (((xx - float(cx)) ** 2 + (yy - float(cy)) ** 2) <= radius**2).to(torch.float32)
        scribble[0] = torch.maximum(scribble[0], patch)
    return (scribble * region_mask).clamp(0.0, 1.0)


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


def _make_target(
    *,
    source: torch.Tensor,
    scribble_mask: torch.Tensor,
    generator: torch.Generator,
) -> torch.Tensor:
    edited = 0.55 * torch.flip(source, dims=(2,)) + 0.45 * torch.roll(source, shifts=2, dims=1)
    edited = 0.10 + 0.90 * edited
    target = source * (1.0 - scribble_mask) + edited * scribble_mask
    target += 0.02 * torch.randn(source.shape, generator=generator, dtype=torch.float32)
    target += 0.01 * torch.rand(source.shape, generator=generator, dtype=torch.float32)
    return target.clamp(0.0, 1.0)


def _make_synthetic_scribble_mask_editing_data(cfg: DataConfig) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(int(cfg.seed))
    sources: list[torch.Tensor] = []
    scribble_masks: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []

    for _ in range(int(cfg.num_samples)):
        region_mask = _region_mask(int(cfg.image_size), generator)
        scribble_mask = _scribble_mask(region_mask, generator)
        source = _make_source(region_mask, generator)
        target = _make_target(source=source, scribble_mask=scribble_mask, generator=generator)
        sources.append(source)
        scribble_masks.append(scribble_mask)
        targets.append(target)

    return (
        torch.stack(sources, dim=0),
        torch.stack(scribble_masks, dim=0),
        torch.stack(targets, dim=0),
    )


class SyntheticScribbleMaskEditingDataset(Dataset):
    def __init__(self, cfg: DataConfig) -> None:
        self.cfg = cfg
        self._source, self._scribble_mask, self._target = _make_synthetic_scribble_mask_editing_data(cfg)

    def __len__(self) -> int:
        return int(self._target.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        i = int(idx)
        return self._source[i], self._scribble_mask[i], self._target[i]


def get_dataloaders(cfg: DataConfig) -> tuple[DataLoader, DataLoader]:
    dataset: Dataset = SyntheticScribbleMaskEditingDataset(cfg)
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


__all__ = ["DataConfig", "SyntheticScribbleMaskEditingDataset", "get_dataloaders"]
