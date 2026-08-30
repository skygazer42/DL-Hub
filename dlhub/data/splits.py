from dataclasses import dataclass
import operator

import numpy as np


@dataclass(frozen=True)
class SplitIndices:
    train: list[int]
    val: list[int]


def _integer(name: str, value: int) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        return operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc


def train_val_split_indices(
    *, n: int, val_fraction: float = 0.2, seed: int = 0
) -> tuple[list[int], list[int]]:
    """Return deterministic train/val indices for a dataset of size `n`.

    This helper is intentionally NumPy-only so it works even when torch isn't installed.
    """

    n = _integer("n", n)
    if n < 2:
        raise ValueError(f"n must contain at least 2 samples, got {n}")

    val_fraction = float(val_fraction)
    if not np.isfinite(val_fraction) or not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")

    seed = _integer("seed", seed)
    if seed < 0:
        raise ValueError(f"seed must be >= 0, got {seed}")

    rng = np.random.default_rng(seed)
    indices = np.arange(n, dtype=np.int64)
    rng.shuffle(indices)

    n_val = int(round(n * val_fraction))
    n_val = max(1, min(n - 1, n_val))

    val_idx = indices[:n_val].tolist()
    train_idx = indices[n_val:].tolist()
    return train_idx, val_idx
