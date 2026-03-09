
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
    dataset: str = "fake"  # "fake" | "mnist"
    batch_size: int = 128
    num_workers: int = 0

    num_samples: int = 2048
    seed: int = 0

    data_dir: str = ".data/mnist"
    download: bool = True

    val_fraction: float = 0.0  # GAN lesson doesn't need val; kept for consistency.


def _make_fake_images(cfg: DataConfig) -> torch.Tensor:
    g = torch.Generator().manual_seed(int(cfg.seed))
    images = torch.rand((int(cfg.num_samples), 1, 28, 28), generator=g, dtype=torch.float32)
    return images


class _TorchvisionImageOnly(Dataset):
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
    except Exception as exc:  # pragma: no cover
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


def get_dataloader(cfg: DataConfig) -> DataLoader:
    if cfg.dataset == "fake":
        images = _make_fake_images(cfg)
        base_ds: Dataset = _ImageOnlyDataset(images)
    elif cfg.dataset == "mnist":
        base_ds = _TorchvisionImageOnly(_make_mnist_dataset(cfg))
    else:
        raise ValueError(f"Unknown dataset: {cfg.dataset!r}")

    # Optionally subsample for fast experimentation.
    if cfg.val_fraction > 0.0:
        train_idx, _ = train_val_split_indices(n=len(base_ds), val_fraction=cfg.val_fraction, seed=cfg.seed)
        base_ds = Subset(base_ds, train_idx)

    ds = base_ds
    loader = DataLoader(
        ds,
        batch_size=int(cfg.batch_size),
        shuffle=True,
        num_workers=int(cfg.num_workers),
        drop_last=True,
    )
    return loader
