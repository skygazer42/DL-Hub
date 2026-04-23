from dataclasses import dataclass
from typing import Any
import numpy as np

from .splits import train_val_split_indices


@dataclass(frozen=True)
class ToyClassificationConfig:
    num_samples: int = 512
    num_features: int = 2
    noise_std: float = 0.2
    val_fraction: float = 0.2
    seed: int = 0


def make_linearly_separable_classification_numpy(
    config: ToyClassificationConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a small linearly-separable binary classification dataset (NumPy).

    Returns:
      x: float64, shape (N, D)
      y: int64, shape (N,)
    """

    rng = np.random.default_rng(int(config.seed))
    x = rng.normal(size=(int(config.num_samples), int(config.num_features))).astype(np.float64)

    # Random separating hyperplane.
    w = rng.normal(size=(x.shape[1],)).astype(np.float64)
    b = float(rng.normal())

    logits = x @ w + b + rng.normal(scale=float(config.noise_std), size=(x.shape[0],))
    y = (logits > 0).astype(np.int64)
    return x, y


def make_toy_classification_dataloaders(
    config: ToyClassificationConfig,
    *,
    batch_size: int = 64,
    num_workers: int = 0,
) -> tuple[Any, Any]:
    """Return torch DataLoaders for the toy classification dataset.

    Torch is imported lazily so that non-torch parts of the repo can still run.
    """

    import torch
    from torch.utils.data import DataLoader, TensorDataset

    x_np, y_np = make_linearly_separable_classification_numpy(config)
    x = torch.from_numpy(x_np).to(torch.float32)
    y = torch.from_numpy(y_np).to(torch.long)

    train_idx, val_idx = train_val_split_indices(
        n=int(config.num_samples), val_fraction=float(config.val_fraction), seed=int(config.seed)
    )

    train_ds = TensorDataset(x[train_idx], y[train_idx])
    val_ds = TensorDataset(x[val_idx], y[val_idx])

    train_loader = DataLoader(
        train_ds,
        batch_size=int(batch_size),
        shuffle=True,
        num_workers=int(num_workers),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
    )
    return train_loader, val_loader
