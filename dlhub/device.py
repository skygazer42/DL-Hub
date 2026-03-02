from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class DeviceInfo:
    name: str
    torch_device: torch.device


def resolve_device(device: str = "auto") -> DeviceInfo:
    """Resolve device string into a torch.device.

    Supported values:
    - auto (default): cuda -> mps -> cpu
    - cpu
    - cuda
    - mps
    - cuda:0 / cuda:1 / ...
    """

    import torch

    requested = (device or "auto").lower()

    if requested == "auto":
        if torch.cuda.is_available():
            return DeviceInfo(name="cuda", torch_device=torch.device("cuda:0"))
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return DeviceInfo(name="mps", torch_device=torch.device("mps"))
        return DeviceInfo(name="cpu", torch_device=torch.device("cpu"))

    if requested == "cpu":
        return DeviceInfo(name="cpu", torch_device=torch.device("cpu"))

    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
        return DeviceInfo(name=requested, torch_device=torch.device(requested))

    if requested == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError("MPS was requested but torch.backends.mps.is_available() is False.")
        return DeviceInfo(name="mps", torch_device=torch.device("mps"))

    raise ValueError(f"Unsupported device: {device!r}")
