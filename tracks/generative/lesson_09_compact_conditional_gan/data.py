from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from dlhub.data.splits import train_val_split_indices


class _ConditionedSyntheticDataset(Dataset):
    def __init__(self, images: torch.Tensor, labels: torch.Tensor) -> None:
        self.images = images
        self.labels = labels

    def __len__(self) -> int:
        return int(self.images.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        i = int(idx)
        return self.images[i], self.labels[i]


@dataclass(frozen=True)
class DataConfig:
    batch_size: int = 32
    image_size: int = 28
    num_classes: int = 4
    num_workers: int = 0
    num_samples: int = 256
    seed: int = 0
    val_fraction: float = 0.2


def _make_class_pattern(image: torch.Tensor, label: int) -> None:
    if label == 0:
        image[4:24, 12:16] = 0.95
    elif label == 1:
        image[12:16, 4:24] = 0.95
    elif label == 2:
        diag = torch.arange(4, 24)
        image[diag, diag] = 0.95
    else:
        diag = torch.arange(4, 24)
        image[diag, 27 - diag] = 0.95


def _make_synthetic_images_and_labels(cfg: DataConfig) -> tuple[torch.Tensor, torch.Tensor]:
    size = int(cfg.image_size)
    if size != 28:
        raise ValueError(f"Synthetic conditional GAN expects image_size=28, got {size}")
    if int(cfg.num_classes) <= 0:
        raise ValueError(f"num_classes must be positive, got {cfg.num_classes}")

    num_samples = int(cfg.num_samples)
    num_classes = int(cfg.num_classes)
    g = torch.Generator().manual_seed(int(cfg.seed))

    labels = torch.randint(0, num_classes, (num_samples,), generator=g, dtype=torch.int64)
    images = torch.zeros((num_samples, 1, size, size), dtype=torch.float32)

    for idx in range(num_samples):
        label = int(labels[idx].item())
        canvas = images[idx, 0]
        _make_class_pattern(canvas, label % 4)
        canvas += 0.08 * torch.rand((size, size), generator=g, dtype=torch.float32)
        canvas.clamp_(0.0, 1.0)

    return images, labels


def get_dataloaders(cfg: DataConfig) -> tuple[DataLoader, DataLoader]:
    images, labels = _make_synthetic_images_and_labels(cfg)
    base_ds: Dataset = _ConditionedSyntheticDataset(images, labels)
    train_idx, val_idx = train_val_split_indices(
        n=len(base_ds), val_fraction=cfg.val_fraction, seed=cfg.seed
    )
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
