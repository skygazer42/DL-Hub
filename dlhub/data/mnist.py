from __future__ import annotations

import torch
from torch.utils.data import DataLoader


def get_mnist_dataloaders(
    *,
    dataset: str = "mnist",
    data_dir: str = ".data",
    batch_size: int = 64,
    num_workers: int = 2,
    resize_to: int | None = None,
) -> tuple[DataLoader, DataLoader]:
    """Return train/test dataloaders for MNIST or a FakeData stand-in.

    dataset:
      - "mnist": downloads MNIST to `data_dir`
      - "fake": uses torchvision FakeData (no downloads) with MNIST-like shapes
    """

    try:
        import torchvision
        from torchvision import transforms
    except Exception as exc:
        raise RuntimeError(
            "torchvision is required for MNIST dataloaders. Install it for the vision track."
        ) from exc

    dataset = dataset.lower().strip()

    parts = []
    if resize_to is not None:
        parts.append(transforms.Resize(int(resize_to)))
    parts.append(transforms.ToTensor())
    parts.append(transforms.Normalize((0.1307,), (0.3081,)))
    transform = transforms.Compose(parts)

    if dataset == "mnist":
        train_ds = torchvision.datasets.MNIST(
            root=data_dir, train=True, transform=transform, download=True
        )
        test_ds = torchvision.datasets.MNIST(
            root=data_dir, train=False, transform=transform, download=True
        )
    elif dataset == "fake":
        train_ds = torchvision.datasets.FakeData(
            size=512,
            image_size=(1, 28, 28),
            num_classes=10,
            transform=transform,
        )
        test_ds = torchvision.datasets.FakeData(
            size=256,
            image_size=(1, 28, 28),
            num_classes=10,
            transform=transform,
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset!r} (expected 'mnist' or 'fake')")

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, test_loader
