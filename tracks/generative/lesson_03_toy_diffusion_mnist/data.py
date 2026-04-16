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


class _TorchvisionImageOnly(Dataset):
    def __init__(self, ds: Dataset) -> None:
        self.ds = ds

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> torch.Tensor:
        image, _ = self.ds[int(idx)]
        return image


@dataclass(frozen=True)
class DataConfig:
    dataset: str = "fake"  # "fake" | "mnist"
    batch_size: int = 128
    num_workers: int = 0

    num_samples: int = 2048
    seed: int = 0

    data_dir: str = ".data/mnist"
    download: bool = True

    val_fraction: float = 0.1


def _make_fake_images(cfg: DataConfig) -> torch.Tensor:
    g = torch.Generator().manual_seed(int(cfg.seed))
    coords = torch.linspace(-1.0, 1.0, 28, dtype=torch.float32)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")

    def sample_uniform(low: float, high: float) -> float:
        return float(low + (high - low) * torch.rand((1,), generator=g).item())

    images = []
    for _ in range(int(cfg.num_samples)):
        canvas = torch.zeros((28, 28), dtype=torch.float32)
        num_blobs = int(torch.randint(2, 5, (1,), generator=g).item())
        for _blob_idx in range(num_blobs):
            cx = sample_uniform(-0.6, 0.6)
            cy = sample_uniform(-0.6, 0.6)
            sx = sample_uniform(0.08, 0.28)
            sy = sample_uniform(0.08, 0.28)
            amp = sample_uniform(0.4, 1.0)
            blob = torch.exp(-((xx - cx) ** 2 / (2 * sx * sx) + (yy - cy) ** 2 / (2 * sy * sy)))
            canvas = torch.maximum(canvas, blob * amp)

        canvas = torch.clamp(canvas + 0.05 * torch.rand((28, 28), generator=g), 0.0, 1.0)
        images.append(canvas.unsqueeze(0))

    return torch.stack(images, dim=0)


def _make_mnist_dataset(cfg: DataConfig) -> Dataset:
    try:
        from torchvision import datasets, transforms
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "torchvision is required for --dataset mnist. "
            "Install it or use --dataset fake."
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
        base_ds: Dataset = _ImageOnlyDataset(_make_fake_images(cfg))
    elif cfg.dataset == "mnist":
        base_ds = _TorchvisionImageOnly(_make_mnist_dataset(cfg))
    else:
        raise ValueError(f"Unknown dataset: {cfg.dataset!r}")

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
