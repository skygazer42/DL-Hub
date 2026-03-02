from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from dlhub.data.splits import train_val_split_indices


class _ImageOnlyDataset(Dataset):
    def __init__(self, images: torch.Tensor) -> None:
        self.images = images

    def __len__(self) -> int:  # noqa: D401 - trivial
        return int(self.images.shape[0])

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.images[int(idx)]


@dataclass(frozen=True)
class DataConfig:
    dataset: str = "fake"  # "fake" | "mnist"
    batch_size: int = 128
    num_workers: int = 0

    # Only used for fake dataset.
    num_samples: int = 2048
    seed: int = 0

    # Only used for MNIST dataset.
    data_dir: str = ".data/mnist"
    download: bool = True

    val_fraction: float = 0.1


def _make_fake_images(cfg: DataConfig) -> torch.Tensor:
    g = torch.Generator().manual_seed(int(cfg.seed))
    images = torch.rand((int(cfg.num_samples), 1, 28, 28), generator=g, dtype=torch.float32)
    return images


class _TorchvisionImageOnly(Dataset):
    """Wrap a torchvision dataset returning (image, label) to return only images."""

    def __init__(self, ds: Dataset) -> None:
        self.ds = ds

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> torch.Tensor:
        image, _ = self.ds[int(idx)]
        return image


def _make_mnist_dataset(cfg: DataConfig) -> Dataset:
    try:
        from torchvision import datasets, transforms
    except Exception as exc:  # pragma: no cover - only hit when torchvision missing
        raise RuntimeError(
            "torchvision is required for --dataset mnist. "
            "Install it (e.g. requirements-vision.txt) or use --dataset fake."
        ) from exc

    ds = datasets.MNIST(
        root=str(cfg.data_dir),
        train=True,
        download=bool(cfg.download),
        transform=transforms.ToTensor(),
    )
    return ds


def get_dataloaders(cfg: DataConfig) -> tuple[DataLoader, DataLoader]:
    if cfg.dataset == "fake":
        images = _make_fake_images(cfg)
        base_ds: Dataset = _ImageOnlyDataset(images)
    elif cfg.dataset == "mnist":
        mnist = _make_mnist_dataset(cfg)
        base_ds = _TorchvisionImageOnly(mnist)
    else:
        raise ValueError(f"Unknown dataset: {cfg.dataset!r}")

    n = len(base_ds)
    train_idx, val_idx = train_val_split_indices(
        n=int(n), val_fraction=cfg.val_fraction, seed=cfg.seed
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
