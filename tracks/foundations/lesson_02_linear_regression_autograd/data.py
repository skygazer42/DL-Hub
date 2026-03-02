from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, TensorDataset


@dataclass(frozen=True)
class DataConfig:
    num_samples: int = 2048
    batch_size: int = 128
    noise_std: float = 0.1


def make_regression_dataloaders(config: DataConfig) -> tuple[DataLoader, DataLoader]:
    rng = torch.Generator().manual_seed(0)

    x = torch.randn(config.num_samples, 2, generator=rng)
    w = torch.tensor([[3.0], [-2.0]])
    b = 0.5
    y = x @ w + b
    # NOTE: `torch.randn_like(..., generator=...)` is not supported on all torch versions.
    noise = torch.randn(y.shape, generator=rng, dtype=y.dtype, device=y.device) * float(
        config.noise_std
    )
    y = y + noise

    n_train = int(0.8 * config.num_samples)
    train_ds = TensorDataset(x[:n_train], y[:n_train])
    test_ds = TensorDataset(x[n_train:], y[n_train:])

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False)
    return train_loader, test_loader
