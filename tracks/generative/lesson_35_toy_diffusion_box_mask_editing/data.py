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


def _box_mask(image_size: int, generator: torch.Generator) -> torch.Tensor:
    mask = torch.zeros((1, image_size, image_size), dtype=torch.float32)
    height = int(torch.randint(image_size // 4, image_size // 2, (1,), generator=generator).item())
    width = int(torch.randint(image_size // 4, image_size // 2, (1,), generator=generator).item())
    y1 = int(torch.randint(1, image_size - height - 1, (1,), generator=generator).item())
    x1 = int(torch.randint(1, image_size - width - 1, (1,), generator=generator).item())
    mask[:, y1 : y1 + height, x1 : x1 + width] = 1.0
    return mask


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


def _make_source(box_mask: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    base = _texture(box_mask.shape[-1], generator)
    background = 0.05 + 0.07 * torch.rand(box_mask.shape, generator=generator, dtype=torch.float32)
    source = background * (1.0 - box_mask) + (0.18 + 0.72 * base) * box_mask
    source += 0.02 * torch.rand(box_mask.shape, generator=generator, dtype=torch.float32)
    return source.clamp(0.0, 1.0)


def _make_target(
    *,
    source: torch.Tensor,
    box_mask: torch.Tensor,
    generator: torch.Generator,
) -> torch.Tensor:
    edited = 0.45 * torch.flip(source, dims=(2,)) + 0.55 * torch.roll(source, shifts=2, dims=1)
    edited = 0.10 + 0.90 * edited
    target = source * (1.0 - box_mask) + edited * box_mask
    target += 0.02 * torch.randn(source.shape, generator=generator, dtype=torch.float32)
    target += 0.01 * torch.rand(source.shape, generator=generator, dtype=torch.float32)
    return target.clamp(0.0, 1.0)


def _make_toy_box_mask_editing_data(cfg: DataConfig) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(int(cfg.seed))
    sources: list[torch.Tensor] = []
    box_masks: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []

    for _ in range(int(cfg.num_samples)):
        box_mask = _box_mask(int(cfg.image_size), generator)
        source = _make_source(box_mask, generator)
        target = _make_target(source=source, box_mask=box_mask, generator=generator)
        sources.append(source)
        box_masks.append(box_mask)
        targets.append(target)

    return (
        torch.stack(sources, dim=0),
        torch.stack(box_masks, dim=0),
        torch.stack(targets, dim=0),
    )


class ToyBoxMaskEditingDataset(Dataset):
    def __init__(self, cfg: DataConfig) -> None:
        self.cfg = cfg
        self._source, self._box_mask, self._target = _make_toy_box_mask_editing_data(cfg)

    def __len__(self) -> int:
        return int(self._target.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        i = int(idx)
        return self._source[i], self._box_mask[i], self._target[i]


def get_dataloaders(cfg: DataConfig) -> tuple[DataLoader, DataLoader]:
    dataset: Dataset = ToyBoxMaskEditingDataset(cfg)
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


__all__ = ["DataConfig", "ToyBoxMaskEditingDataset", "get_dataloaders"]
