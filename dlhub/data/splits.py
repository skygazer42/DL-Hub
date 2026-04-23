from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class SplitIndices:
    train: list[int]
    val: list[int]


def train_val_split_indices(
    *, n: int, val_fraction: float = 0.2, seed: int = 0
) -> tuple[list[int], list[int]]:
    """Return deterministic train/val indices for a dataset of size `n`.

    This helper is intentionally NumPy-only so it works even when torch isn't installed.
    """

    n = int(n)
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")

    val_fraction = float(val_fraction)
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")

    rng = np.random.default_rng(int(seed))
    indices = np.arange(n, dtype=np.int64)
    rng.shuffle(indices)

    n_val = int(round(n * val_fraction))
    n_val = max(1, min(n - 1, n_val))

    val_idx = indices[:n_val].tolist()
    train_idx = indices[n_val:].tolist()
    return train_idx, val_idx
