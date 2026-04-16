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
        cx = float(torch.empty((1,)).uniform_(-0.4, 0.4, generator=generator).item())
        cy = float(torch.empty((1,)).uniform_(-0.4, 0.4, generator=generator).item())
        radius = float(torch.empty((1,)).uniform_(0.24, 0.48, generator=generator).item())
        fg = (xx - cx).pow(2) + (yy - cy).pow(2) <= radius**2
        mask[0, fg] = 1.0
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
        slope = float(torch.empty((1,)).uniform_(-0.9, 0.9, generator=generator).item())
        bias = float(torch.empty((1,)).uniform_(-0.35, 0.35, generator=generator).item())
        width = float(torch.empty((1,)).uniform_(0.1, 0.24, generator=generator).item())
        diag = torch.abs(yy - slope * xx - bias) <= width
        mask[0, diag] = 1.0
    return mask


def _reference_texture(image_size: int, generator: torch.Generator) -> torch.Tensor:
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, image_size, dtype=torch.float32),
        torch.linspace(-1.0, 1.0, image_size, dtype=torch.float32),
        indexing="ij",
    )
    mode = int(torch.randint(0, 3, (1,), generator=generator).item())
    if mode == 0:
        phase = float(torch.empty((1,)).uniform_(-3.14, 3.14, generator=generator).item())
        ref = 0.5 + 0.5 * torch.sin(xx * 8.0 + phase)
    elif mode == 1:
        ref = (((xx * 6.0).floor() + (yy * 6.0).floor()) % 2.0 == 0.0).to(torch.float32)
    else:
        ref = 0.5 + 0.5 * torch.cos((xx.pow(2) + yy.pow(2)).sqrt() * 10.0)
    ref = ref.unsqueeze(0)
    ref += 0.05 * torch.rand((1, image_size, image_size), generator=generator, dtype=torch.float32)
    return ref.clamp(0.0, 1.0)


def _condition_map(mask: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    blurred = 0.15 + 0.7 * mask
    blurred += 0.08 * torch.rand(mask.shape, generator=generator, dtype=torch.float32)
    return blurred.clamp(0.0, 1.0)


def _compose_target(
    *,
    mask: torch.Tensor,
    reference: torch.Tensor,
    condition: torch.Tensor,
    generator: torch.Generator,
) -> torch.Tensor:
    background = 0.05 + 0.08 * torch.rand(mask.shape, generator=generator, dtype=torch.float32)
    guided_foreground = 0.72 * reference + 0.28 * condition
    target = background * (1.0 - mask) + guided_foreground * mask
    target += 0.03 * torch.rand(mask.shape, generator=generator, dtype=torch.float32)
    return target.clamp(0.0, 1.0)


class ToyReferenceGuidedGenerationDataset(Dataset):
    def __init__(self, cfg: DataConfig) -> None:
        self.cfg = cfg
        self._reference, self._condition, self._target = _make_toy_reference_guided_data(cfg)

    def __len__(self) -> int:
        return int(self._target.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        i = int(idx)
        return self._reference[i], self._condition[i], self._target[i]


def _make_toy_reference_guided_data(cfg: DataConfig) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(int(cfg.seed))
    references: list[torch.Tensor] = []
    conditions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []

    for _ in range(int(cfg.num_samples)):
        mask = _shape_mask(int(cfg.image_size), generator)
        reference = _reference_texture(int(cfg.image_size), generator)
        condition = _condition_map(mask, generator)
        target = _compose_target(mask=mask, reference=reference, condition=condition, generator=generator)
        references.append(reference)
        conditions.append(condition)
        targets.append(target)

    return torch.stack(references, dim=0), torch.stack(conditions, dim=0), torch.stack(targets, dim=0)


def get_dataloaders(cfg: DataConfig) -> tuple[DataLoader, DataLoader]:
    dataset: Dataset = ToyReferenceGuidedGenerationDataset(cfg)
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


__all__ = ["DataConfig", "ToyReferenceGuidedGenerationDataset", "get_dataloaders"]
