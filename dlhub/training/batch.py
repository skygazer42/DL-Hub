from __future__ import annotations

from collections import defaultdict
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

    Common mapping and named-tuple types are preserved when constructible. Any
    non-tensor leaf values are returned unchanged.
    """

    import torch

    if torch.is_tensor(batch):
        return batch.to(device)

    if isinstance(batch, Mapping):
        moved = {k: to_device(v, device=device) for k, v in batch.items()}
        if type(batch) is dict:
            return moved
        if isinstance(batch, defaultdict):
            return type(batch)(batch.default_factory, moved)
        try:
            return type(batch)(moved)
        except (TypeError, ValueError):
            # Some read-only/custom mappings cannot be reconstructed. Falling
            # back to dict preserves the historical return contract.
            return moved

    if isinstance(batch, tuple):
        moved = tuple(to_device(v, device=device) for v in batch)
        if hasattr(batch, "_fields"):
            try:
                return type(batch)(*moved)
            except TypeError:
                pass
        return moved

    if isinstance(batch, list):
        return [to_device(v, device=device) for v in batch]

    # NOTE: We intentionally don't try to handle arbitrary Sequences (like strings).
    return batch
