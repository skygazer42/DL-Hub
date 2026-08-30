from __future__ import annotations

import pickle
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ._atomic import atomic_write

if TYPE_CHECKING:
    import torch


def save_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    epoch: int | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Save a lightweight training checkpoint.

    Torch is imported lazily to keep repo-root utilities usable without torch.
    """

    import torch

    out_path = Path(path)

    payload: dict[str, Any] = {
        "model_state": model.state_dict(),
        "epoch": int(epoch) if epoch is not None else None,
        "extra": dict(extra or {}),
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()

    def write(handle) -> None:
        # State-dict serialization is intentional; every library read uses the restricted loader.
        # nosemgrep: trailofbits.python.pickles-in-pytorch.pickles-in-pytorch
        torch.save(payload, handle)

    return atomic_write(out_path, write)


def load_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device | None = "cpu",
    allow_unsafe_legacy: bool = False,
) -> dict[str, Any]:
    """Load a checkpoint saved by :func:`save_checkpoint` and restore state.

    Checkpoints are loaded with PyTorch's restricted ``weights_only`` loader by
    default. ``allow_unsafe_legacy`` exists only for trusted legacy checkpoints:
    unrestricted pickle loading can execute arbitrary code embedded in a file.
    """

    import torch

    ckpt_path = Path(path)
    try:
        # This is the restricted PyTorch loader, which the generic rule does not distinguish.
        # nosemgrep: trailofbits.python.pickles-in-pytorch.pickles-in-pytorch
        payload = torch.load(ckpt_path, map_location=map_location, weights_only=True)
    except (TypeError, pickle.UnpicklingError) as exc:
        if not allow_unsafe_legacy:
            raise RuntimeError(
                "Safe checkpoint loading failed; refusing to retry with unrestricted pickle. "
                "Upgrade PyTorch or, only for a checkpoint from a trusted source, pass "
                "allow_unsafe_legacy=True."
            ) from exc

        warnings.warn(
            "Unsafe legacy checkpoint loading uses unrestricted pickle and can execute "
            "arbitrary code. Continue only with a checkpoint you trust.",
            RuntimeWarning,
            stacklevel=2,
        )
        # Intentional trusted-only escape hatch, gated by allow_unsafe_legacy and the warning above.
        # nosemgrep: trailofbits.python.pickles-in-pytorch.pickles-in-pytorch
        payload = torch.load(ckpt_path, map_location=map_location)

    if not isinstance(payload, Mapping):
        raise TypeError(f"Checkpoint payload must be a mapping, got {type(payload).__name__}")
    if "model_state" not in payload:
        raise KeyError("Checkpoint payload is missing required key 'model_state'")

    model.load_state_dict(payload["model_state"])
    if optimizer is not None and "optimizer_state" in payload:
        optimizer.load_state_dict(payload["optimizer_state"])

    return {
        "epoch": payload.get("epoch"),
        "extra": payload.get("extra", {}),
    }
