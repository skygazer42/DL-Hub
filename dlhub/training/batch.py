from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


def to_device(batch: Any, *, device: torch.device) -> Any:
    """Recursively move a nested batch to a torch device.

    Supports:
    - torch.Tensor
    - dict-like mappings
    - lists / tuples

    Any non-tensor leaf values are returned unchanged.
    """

    import torch

    if torch.is_tensor(batch):
        return batch.to(device)

    if isinstance(batch, Mapping):
        return {k: to_device(v, device=device) for k, v in batch.items()}

    if isinstance(batch, tuple):
        return tuple(to_device(v, device=device) for v in batch)

    if isinstance(batch, list):
        return [to_device(v, device=device) for v in batch]

    # NOTE: We intentionally don't try to handle arbitrary Sequences (like strings).
    return batch
