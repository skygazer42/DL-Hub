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


def _texture(image_size: int, generator: torch.Generator) -> torch.Tensor:
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, image_size, dtype=torch.float32),
        torch.linspace(-1.0, 1.0, image_size, dtype=torch.float32),
        indexing="ij",
    )
    mode = int(torch.randint(0, 5, (1,), generator=generator).item())
    if mode == 0:
        phase = float(torch.empty((1,)).uniform_(-3.14, 3.14, generator=generator).item())
        image = 0.5 + 0.5 * torch.sin(xx * 8.0 + yy * 2.5 + phase)
    elif mode == 1:
        image = 0.5 + 0.5 * torch.cos(yy * 9.5)
    elif mode == 2:
        image = (((xx * 7.0).floor() + (yy * 7.0).floor()) % 2.0 == 0.0).to(torch.float32)
    elif mode == 3:
        radial = (xx.pow(2) + yy.pow(2)).sqrt()
        image = 0.5 + 0.5 * torch.cos(radial * 11.0)
    else:
        image = 0.5 + 0.5 * torch.sin((xx + yy) * 6.0) * torch.cos((xx - yy) * 4.0)
    image = image.unsqueeze(0)
    image += 0.05 * torch.rand((1, image_size, image_size), generator=generator, dtype=torch.float32)
    return image.clamp(0.0, 1.0)


def _make_layout(image_size: int, generator: torch.Generator) -> torch.Tensor:
    base = _shape_mask(image_size, generator)
    accent = _shape_mask(image_size, generator)
    contour = 0.65 * base + 0.35 * accent
    contour += 0.04 * torch.rand(contour.shape, generator=generator, dtype=torch.float32)
    return contour.clamp(0.0, 1.0)


def _make_target(layout: torch.Tensor, reference: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    structure = (layout > 0.28).to(layout.dtype)
    shifted_reference = torch.roll(reference, shifts=1, dims=2)
    alt_reference = torch.roll(reference, shifts=-1, dims=1)
    fused_reference = 0.58 * shifted_reference + 0.42 * (1.0 - alt_reference)
    foreground = (0.18 + 0.72 * fused_reference) * structure
    background = (0.05 + 0.18 * (1.0 - layout)) * (1.0 - structure)
    target = foreground + background
    target = 0.72 * target + 0.28 * layout
    target += 0.02 * torch.randn(layout.shape, generator=generator, dtype=torch.float32)
    target += 0.01 * torch.rand(layout.shape, generator=generator, dtype=torch.float32)
    return target.clamp(0.0, 1.0)


def _make_synthetic_layout_reference_fusion_data(cfg: DataConfig) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(int(cfg.seed))
    layouts: list[torch.Tensor] = []
    references: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []

    for _ in range(int(cfg.num_samples)):
        layout = _make_layout(int(cfg.image_size), generator)
        reference = _texture(int(cfg.image_size), generator)
        target = _make_target(layout, reference, generator)
        layouts.append(layout)
        references.append(reference)
        targets.append(target)

    return torch.stack(layouts, dim=0), torch.stack(references, dim=0), torch.stack(targets, dim=0)


class SyntheticLayoutReferenceFusionDataset(Dataset):
    def __init__(self, cfg: DataConfig) -> None:
        self.cfg = cfg
        self._layout, self._reference, self._target = _make_synthetic_layout_reference_fusion_data(cfg)

    def __len__(self) -> int:
        return int(self._target.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        i = int(idx)
        return self._layout[i], self._reference[i], self._target[i]


def get_dataloaders(cfg: DataConfig) -> tuple[DataLoader, DataLoader]:
    dataset: Dataset = SyntheticLayoutReferenceFusionDataset(cfg)
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


__all__ = ["DataConfig", "SyntheticLayoutReferenceFusionDataset", "get_dataloaders"]
