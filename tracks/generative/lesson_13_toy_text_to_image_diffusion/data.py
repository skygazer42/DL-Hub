from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from dlhub.data.splits import train_val_split_indices

PROMPTS = (
    "vertical bar",
    "horizontal bar",
    "square blob",
    "ring blob",
)


@dataclass(frozen=True)
class DataConfig:
    batch_size: int = 32
    image_size: int = 28
    num_workers: int = 0
    num_samples: int = 256
    seed: int = 0
    val_fraction: float = 0.2


class _ToyTextImageDataset(Dataset):
    def __init__(self, token_ids: torch.Tensor, images: torch.Tensor) -> None:
        self.token_ids = token_ids
        self.images = images

    def __len__(self) -> int:
        return int(self.token_ids.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        i = int(idx)
        return self.token_ids[i], self.images[i]


def _render_scene(token_id: int, image_size: int, g: torch.Generator) -> torch.Tensor:
    image = torch.zeros((1, image_size, image_size), dtype=torch.float32)
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, image_size, dtype=torch.float32),
        torch.linspace(-1.0, 1.0, image_size, dtype=torch.float32),
        indexing="ij",
    )

    if token_id == 0:
        x_center = int(torch.randint(low=6, high=image_size - 6, size=(1,), generator=g).item())
        width = int(torch.randint(low=2, high=5, size=(1,), generator=g).item())
        x0 = max(0, x_center - width)
        x1 = min(image_size, x_center + width)
        image[:, :, x0:x1] = 0.9
    elif token_id == 1:
        y_center = int(torch.randint(low=6, high=image_size - 6, size=(1,), generator=g).item())
        width = int(torch.randint(low=2, high=5, size=(1,), generator=g).item())
        y0 = max(0, y_center - width)
        y1 = min(image_size, y_center + width)
        image[:, y0:y1, :] = 0.85
    elif token_id == 2:
        side = int(torch.randint(low=7, high=13, size=(1,), generator=g).item())
        x0 = int(torch.randint(low=2, high=image_size - side - 1, size=(1,), generator=g).item())
        y0 = int(torch.randint(low=2, high=image_size - side - 1, size=(1,), generator=g).item())
        image[:, y0 : y0 + side, x0 : x0 + side] = 0.95
    else:
        cx = float(torch.empty((1,)).uniform_(-0.4, 0.4, generator=g).item())
        cy = float(torch.empty((1,)).uniform_(-0.4, 0.4, generator=g).item())
        radius = float(torch.empty((1,)).uniform_(0.25, 0.5, generator=g).item())
        thickness = float(torch.empty((1,)).uniform_(0.06, 0.12, generator=g).item())
        dist = torch.sqrt((xx - cx).pow(2) + (yy - cy).pow(2))
        ring = (dist >= radius - thickness) & (dist <= radius + thickness)
        image[0, ring] = 0.9

    noise = 0.04 * torch.rand((1, image_size, image_size), generator=g, dtype=torch.float32)
    return (image + noise).clamp_(0.0, 1.0)


def _make_dataset(cfg: DataConfig) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(int(cfg.seed))
    n = int(cfg.num_samples)
    size = int(cfg.image_size)
    vocab_size = len(PROMPTS)

    token_ids = torch.randint(low=0, high=vocab_size, size=(n,), generator=g, dtype=torch.long)
    images = torch.zeros((n, 1, size, size), dtype=torch.float32)
    for i in range(n):
        token = int(token_ids[i].item())
        images[i] = _render_scene(token, size, g)
    return token_ids, images


def get_dataloaders(cfg: DataConfig) -> tuple[DataLoader, DataLoader]:
    token_ids, images = _make_dataset(cfg)
    ds: Dataset = _ToyTextImageDataset(token_ids, images)
    train_idx, val_idx = train_val_split_indices(
        n=len(ds),
        val_fraction=float(cfg.val_fraction),
        seed=int(cfg.seed),
    )

    train_loader = DataLoader(
        Subset(ds, train_idx),
        batch_size=int(cfg.batch_size),
        shuffle=True,
        num_workers=int(cfg.num_workers),
        drop_last=False,
    )
    val_loader = DataLoader(
        Subset(ds, val_idx),
        batch_size=int(cfg.batch_size),
        shuffle=False,
        num_workers=int(cfg.num_workers),
        drop_last=False,
    )
    return train_loader, val_loader
