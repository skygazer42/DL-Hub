from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from dlhub.data.splits import train_val_split_indices


class _ImageOnlyDataset(Dataset):
    def __init__(self, images: torch.Tensor) -> None:
        self.images = images

    def __len__(self) -> int:
        return int(self.images.shape[0])

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.images[int(idx)]


@dataclass(frozen=True)
class DataConfig:
    batch_size: int = 32
    image_size: int = 28
    num_workers: int = 0
    num_samples: int = 256
    seed: int = 0
    val_fraction: float = 0.2


def _make_synthetic_images(cfg: DataConfig) -> torch.Tensor:
    size = int(cfg.image_size)
    g = torch.Generator().manual_seed(int(cfg.seed))
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, size, dtype=torch.float32),
        torch.linspace(-1.0, 1.0, size, dtype=torch.float32),
        indexing="ij",
    )
    images = []
    for _ in range(int(cfg.num_samples)):
        image = torch.zeros((size, size), dtype=torch.float32)
        shape_type = int(torch.randint(0, 4, (1,), generator=g).item())

        if shape_type == 0:
            cx = float(torch.empty((1,)).uniform_(-0.6, 0.6, generator=g).item())
            cy = float(torch.empty((1,)).uniform_(-0.6, 0.6, generator=g).item())
            radius = float(torch.empty((1,)).uniform_(0.2, 0.5, generator=g).item())
            mask = (xx - cx).pow(2) + (yy - cy).pow(2) <= radius**2
            image[mask] = 0.9
        elif shape_type == 1:
            y1 = int(torch.randint(3, size // 2, (1,), generator=g).item())
            y2 = int(torch.randint(size // 2, size - 3, (1,), generator=g).item())
            x1 = int(torch.randint(3, size // 2, (1,), generator=g).item())
            x2 = int(torch.randint(size // 2, size - 3, (1,), generator=g).item())
            image[y1:y2, x1:x2] = 0.95
        elif shape_type == 2:
            center = int(torch.randint(7, size - 7, (1,), generator=g).item())
            thickness = int(torch.randint(1, 3, (1,), generator=g).item())
            image[center - thickness : center + thickness + 1, :] = 0.9
            image[:, center - thickness : center + thickness + 1] = 0.9
        else:
            slope = float(torch.empty((1,)).uniform_(-1.2, 1.2, generator=g).item())
            offset = float(torch.empty((1,)).uniform_(-0.4, 0.4, generator=g).item())
            line = torch.abs(yy - (slope * xx + offset))
            image[line < 0.08] = 0.85

        image = torch.clamp(image + 0.05 * torch.rand((size, size), generator=g), 0.0, 1.0)
        images.append(image.unsqueeze(0))

    return torch.stack(images, dim=0)


def get_dataloaders(cfg: DataConfig) -> tuple[DataLoader, DataLoader]:
    base_ds: Dataset = _ImageOnlyDataset(_make_synthetic_images(cfg))
    train_idx, val_idx = train_val_split_indices(
        n=len(base_ds),
        val_fraction=float(cfg.val_fraction),
        seed=int(cfg.seed),
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
