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


def _draw_rectangle(
    image: torch.Tensor,
    *,
    top: int,
    left: int,
    height: int,
    width: int,
    value: float,
) -> None:
    image[top : top + height, left : left + width] = value


def _make_toy_images(cfg: DataConfig) -> torch.Tensor:
    size = int(cfg.image_size)
    if size != 28:
        raise ValueError(f"Toy latent diffusion lesson expects image_size=28, got {size}")

    g = torch.Generator().manual_seed(int(cfg.seed))
    images = torch.zeros((int(cfg.num_samples), 1, size, size), dtype=torch.float32)

    yy, xx = torch.meshgrid(
        torch.arange(size, dtype=torch.float32),
        torch.arange(size, dtype=torch.float32),
        indexing="ij",
    )

    for idx in range(int(cfg.num_samples)):
        image = images[idx, 0]
        pattern = int(torch.randint(0, 4, (1,), generator=g).item())

        if pattern == 0:
            col = int(torch.randint(4, size - 6, (1,), generator=g).item())
            width = int(torch.randint(3, 6, (1,), generator=g).item())
            _draw_rectangle(image, top=4, left=col, height=size - 8, width=width, value=0.9)
        elif pattern == 1:
            row = int(torch.randint(4, size - 6, (1,), generator=g).item())
            height = int(torch.randint(3, 6, (1,), generator=g).item())
            _draw_rectangle(image, top=row, left=4, height=height, width=size - 8, value=0.85)
        elif pattern == 2:
            center_x = int(torch.randint(8, size - 8, (1,), generator=g).item())
            center_y = int(torch.randint(8, size - 8, (1,), generator=g).item())
            radius = float(torch.randint(4, 7, (1,), generator=g).item())
            mask = (xx - center_x).pow(2) + (yy - center_y).pow(2) <= radius**2
            image[mask] = 1.0
        else:
            offset = int(torch.randint(-2, 3, (1,), generator=g).item())
            diag = torch.arange(size)
            image[diag, torch.clamp(diag + offset, min=0, max=size - 1)] = 0.95
            image[diag, torch.clamp((size - 1 - diag) + offset, min=0, max=size - 1)] = 0.6

        image += 0.05 * torch.rand((size, size), generator=g, dtype=torch.float32)
        image.clamp_(0.0, 1.0)

    return images


def get_dataloaders(cfg: DataConfig) -> tuple[DataLoader, DataLoader]:
    images = _make_toy_images(cfg)
    base_ds: Dataset = _ImageOnlyDataset(images)

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

