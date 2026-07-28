from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from dlhub.data.splits import train_val_split_indices


class _LayoutImageDataset(Dataset):
    def __init__(self, layouts: torch.Tensor, images: torch.Tensor) -> None:
        self.layouts = layouts
        self.images = images

    def __len__(self) -> int:
        return int(self.layouts.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        i = int(idx)
        return self.layouts[i], self.images[i]


@dataclass(frozen=True)
class DataConfig:
    batch_size: int = 32
    image_size: int = 28
    num_classes: int = 4
    num_workers: int = 0
    num_samples: int = 256
    seed: int = 0
    val_fraction: float = 0.2


def _draw_random_rect(layout: torch.Tensor, cls: int, g: torch.Generator) -> None:
    size = int(layout.shape[-1])
    width = int(torch.randint(low=4, high=12, size=(1,), generator=g).item())
    height = int(torch.randint(low=4, high=12, size=(1,), generator=g).item())
    x0 = int(torch.randint(low=0, high=max(1, size - width), size=(1,), generator=g).item())
    y0 = int(torch.randint(low=0, high=max(1, size - height), size=(1,), generator=g).item())
    x1 = min(size, x0 + width)
    y1 = min(size, y0 + height)
    layout[cls, y0:y1, x0:x1] = 1.0


def _make_layout_image_pairs(cfg: DataConfig) -> tuple[torch.Tensor, torch.Tensor]:
    size = int(cfg.image_size)
    num_classes = int(cfg.num_classes)
    if size != 28:
        raise ValueError(f"Synthetic layout-to-image expects image_size=28, got {size}")
    if num_classes <= 0:
        raise ValueError(f"num_classes must be positive, got {num_classes}")

    g = torch.Generator().manual_seed(int(cfg.seed))
    layouts = torch.zeros((int(cfg.num_samples), num_classes, size, size), dtype=torch.float32)
    images = torch.zeros((int(cfg.num_samples), 1, size, size), dtype=torch.float32)

    class_intensity = torch.linspace(0.25, 0.95, steps=num_classes, dtype=torch.float32).view(num_classes, 1, 1)
    for i in range(int(cfg.num_samples)):
        sample_layout = layouts[i]
        num_objects = int(torch.randint(low=1, high=min(4, num_classes + 1), size=(1,), generator=g).item())
        active_classes = torch.randperm(num_classes, generator=g)[:num_objects]
        for cls in active_classes:
            _draw_random_rect(sample_layout, int(cls.item()), g)

        rendered = (sample_layout * class_intensity).amax(dim=0, keepdim=True)
        noise = 0.03 * torch.rand((1, size, size), generator=g, dtype=torch.float32)
        images[i] = (rendered + noise).clamp_(0.0, 1.0)

    return layouts, images


def get_dataloaders(cfg: DataConfig) -> tuple[DataLoader, DataLoader]:
    layouts, images = _make_layout_image_pairs(cfg)
    base_ds: Dataset = _LayoutImageDataset(layouts, images)
    train_idx, val_idx = train_val_split_indices(n=len(base_ds), val_fraction=cfg.val_fraction, seed=cfg.seed)
    train_ds = Subset(base_ds, train_idx)
    val_ds = Subset(base_ds, val_idx)

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg.batch_size),
        shuffle=True,
        num_workers=int(cfg.num_workers),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg.batch_size),
        shuffle=False,
        num_workers=int(cfg.num_workers),
        drop_last=False,
    )
    return train_loader, val_loader
