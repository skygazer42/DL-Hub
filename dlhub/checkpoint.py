from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

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
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "model_state": model.state_dict(),
        "epoch": int(epoch) if epoch is not None else None,
        "extra": dict(extra or {}),
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()

    torch.save(payload, out_path)
    return out_path


def load_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device | None = "cpu",
) -> dict[str, Any]:
    """Load a checkpoint saved by :func:`save_checkpoint` and restore state."""

    import torch

    ckpt_path = Path(path)
    try:
        payload = torch.load(ckpt_path, map_location=map_location, weights_only=True)
    except TypeError:
        payload = torch.load(ckpt_path, map_location=map_location)

    model.load_state_dict(payload["model_state"])
    if optimizer is not None and "optimizer_state" in payload:
        optimizer.load_state_dict(payload["optimizer_state"])

    return {
        "epoch": payload.get("epoch"),
        "extra": payload.get("extra", {}),
    }
