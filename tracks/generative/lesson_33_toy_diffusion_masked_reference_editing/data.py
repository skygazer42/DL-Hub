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


def _shape_mask(image_size: int, generator: torch.Generator) -> torch.Tensor:
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, image_size, dtype=torch.float32),
        torch.linspace(-1.0, 1.0, image_size, dtype=torch.float32),
        indexing="ij",
    )
    mask = torch.zeros((1, image_size, image_size), dtype=torch.float32)
    mode = int(torch.randint(0, 4, (1,), generator=generator).item())

    if mode == 0:
        cx = float(torch.empty((1,)).uniform_(-0.35, 0.35, generator=generator).item())
        cy = float(torch.empty((1,)).uniform_(-0.35, 0.35, generator=generator).item())
        radius = float(torch.empty((1,)).uniform_(0.22, 0.46, generator=generator).item())
        region = (xx - cx).pow(2) + (yy - cy).pow(2) <= radius**2
        mask[0, region] = 1.0
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
    else:
        slope = float(torch.empty((1,)).uniform_(-0.8, 0.8, generator=generator).item())
        bias = float(torch.empty((1,)).uniform_(-0.3, 0.3, generator=generator).item())
        width = float(torch.empty((1,)).uniform_(0.08, 0.18, generator=generator).item())
        diagonal = torch.abs(yy - slope * xx - bias) <= width
        mask[0, diagonal] = 1.0
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
        image = 0.5 + 0.5 * torch.sin(xx * 7.0 + phase)
    elif mode == 1:
        image = 0.5 + 0.5 * torch.cos(yy * 9.5)
    elif mode == 2:
        image = (((xx * 6.0).floor() + (yy * 6.0).floor()) % 2.0 == 0.0).to(torch.float32)
    else:
        image = 0.5 + 0.5 * torch.cos((xx.pow(2) + yy.pow(2)).sqrt() * 10.0)
    image = image.unsqueeze(0)
    image += 0.05 * torch.rand((1, image_size, image_size), generator=generator, dtype=torch.float32)
    return image.clamp(0.0, 1.0)


def _make_source(mask: torch.Tensor, texture: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    background = 0.06 + 0.08 * torch.rand(mask.shape, generator=generator, dtype=torch.float32)
    source = background * (1.0 - mask) + (0.18 + 0.74 * texture) * mask
    source += 0.02 * torch.rand(mask.shape, generator=generator, dtype=torch.float32)
    return source.clamp(0.0, 1.0)


def _make_target(
    *,
    mask: torch.Tensor,
    source: torch.Tensor,
    reference: torch.Tensor,
    generator: torch.Generator,
) -> torch.Tensor:
    shifted_reference = 0.55 * reference + 0.45 * torch.roll(reference, shifts=1, dims=2)
    preserved = source * (1.0 - mask)
    edited = (0.20 * source + 0.80 * shifted_reference) * mask
    target = preserved + edited
    target += 0.02 * torch.randn(source.shape, generator=generator, dtype=torch.float32)
    target += 0.01 * torch.rand(source.shape, generator=generator, dtype=torch.float32)
    return target.clamp(0.0, 1.0)


def _make_toy_masked_reference_editing_data(
    cfg: DataConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(int(cfg.seed))
    sources: list[torch.Tensor] = []
    references: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []

    for _ in range(int(cfg.num_samples)):
        mask = _shape_mask(int(cfg.image_size), generator)
        source_texture = _texture(int(cfg.image_size), generator)
        reference = _texture(int(cfg.image_size), generator)
        source = _make_source(mask, source_texture, generator)
        target = _make_target(mask=mask, source=source, reference=reference, generator=generator)
        sources.append(source)
        references.append(reference)
        masks.append(mask)
        targets.append(target)

    return (
        torch.stack(sources, dim=0),
        torch.stack(references, dim=0),
        torch.stack(masks, dim=0),
        torch.stack(targets, dim=0),
    )


class ToyMaskedReferenceEditingDataset(Dataset):
    def __init__(self, cfg: DataConfig) -> None:
        self.cfg = cfg
        self._source, self._reference, self._mask, self._target = _make_toy_masked_reference_editing_data(cfg)

    def __len__(self) -> int:
        return int(self._target.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        i = int(idx)
        return self._source[i], self._reference[i], self._mask[i], self._target[i]


def get_dataloaders(cfg: DataConfig) -> tuple[DataLoader, DataLoader]:
    dataset: Dataset = ToyMaskedReferenceEditingDataset(cfg)
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


__all__ = ["DataConfig", "ToyMaskedReferenceEditingDataset", "get_dataloaders"]
