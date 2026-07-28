from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from dlhub.data.splits import train_val_split_indices


@dataclass(frozen=True)
class DataConfig:
    batch_size: int = 16
    image_size: int = 32
    num_workers: int = 0
    num_samples: int = 128
    seed: int = 0
    val_fraction: float = 0.2


def _make_image(generator: torch.Generator, image_size: int) -> torch.Tensor:
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, image_size, dtype=torch.float32),
        torch.linspace(-1.0, 1.0, image_size, dtype=torch.float32),
        indexing="ij",
    )
    cx = float(torch.empty((1,)).uniform_(-0.4, 0.4, generator=generator).item())
    cy = float(torch.empty((1,)).uniform_(-0.4, 0.4, generator=generator).item())
    radius = float(torch.empty((1,)).uniform_(0.2, 0.55, generator=generator).item())
    shape = ((xx - cx).pow(2) + (yy - cy).pow(2) <= radius**2).to(torch.float32)

    red = (0.2 + 0.8 * shape).clamp(0.0, 1.0)
    green = (0.1 + 0.7 * torch.roll(shape, shifts=2, dims=0)).clamp(0.0, 1.0)
    blue = (0.1 + 0.7 * torch.roll(shape, shifts=-2, dims=1)).clamp(0.0, 1.0)
    image = torch.stack([red, green, blue], dim=0)
    image += 0.02 * torch.rand((3, image_size, image_size), generator=generator, dtype=torch.float32)
    return image.clamp(0.0, 1.0)


def _make_targets(image: torch.Tensor) -> dict[str, torch.Tensor]:
    pooled = torch.nn.functional.adaptive_avg_pool2d(image.mean(dim=0, keepdim=True), (10, 10))
    density = pooled.unsqueeze(1).repeat(1, 10, 1, 1)

    gx = torch.nn.functional.adaptive_avg_pool2d(image[0:1], (10, 1)).squeeze(-1).squeeze(0)
    gy = torch.nn.functional.adaptive_avg_pool2d(image[1:2], (10, 1)).squeeze(-1).squeeze(0)
    gz = torch.nn.functional.adaptive_avg_pool2d(image[2:3], (10, 1)).squeeze(-1).squeeze(0)
    mesh_tokens = torch.stack([gx, gy, gz], dim=1)
    return {"density": density, "mesh_tokens": mesh_tokens}


class SyntheticImageTo3DDataset(Dataset):
    def __init__(self, cfg: DataConfig) -> None:
        self.cfg = cfg
        generator = torch.Generator().manual_seed(int(cfg.seed))
        images: list[torch.Tensor] = []
        densities: list[torch.Tensor] = []
        meshes: list[torch.Tensor] = []
        for _ in range(int(cfg.num_samples)):
            image = _make_image(generator, int(cfg.image_size))
            targets = _make_targets(image)
            images.append(image)
            densities.append(targets["density"])
            meshes.append(targets["mesh_tokens"])
        self._images = torch.stack(images, dim=0)
        self._density = torch.stack(densities, dim=0)
        self._mesh_tokens = torch.stack(meshes, dim=0)

    def __len__(self) -> int:
        return int(self._images.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        i = int(idx)
        return self._images[i], {"density": self._density[i], "mesh_tokens": self._mesh_tokens[i]}


def get_dataloaders(cfg: DataConfig) -> tuple[DataLoader, DataLoader]:
    dataset: Dataset = SyntheticImageTo3DDataset(cfg)
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


__all__ = ["DataConfig", "SyntheticImageTo3DDataset", "get_dataloaders"]
