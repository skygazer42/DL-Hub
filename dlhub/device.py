from __future__ import annotations

from dataclasses import dataclass
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


_CUDA_DEVICE_PATTERN = re.compile(r"cuda(?::(0|[1-9][0-9]*))?")


@dataclass(frozen=True)
class DeviceInfo:
    name: str
    torch_device: torch.device


def resolve_device(device: str | None = "auto") -> DeviceInfo:
    """Resolve device string into a torch.device.

    Supported values:
    - auto (default): cuda -> mps -> cpu
    - cpu
    - cuda
    - mps
    - cuda:0 / cuda:1 / ...
    """

    import torch

    if device is not None and not isinstance(device, str):
        raise TypeError(f"device must be a string or None, got {type(device).__name__}")
    requested = device.strip().lower() if device else "auto"
    if not requested:
        requested = "auto"

    if requested == "auto":
        if torch.cuda.is_available():
            return DeviceInfo(name="cuda", torch_device=torch.device("cuda:0"))
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return DeviceInfo(name="mps", torch_device=torch.device("mps"))
        return DeviceInfo(name="cpu", torch_device=torch.device("cpu"))

    if requested == "cpu":
        return DeviceInfo(name="cpu", torch_device=torch.device("cpu"))

    cuda_match = _CUDA_DEVICE_PATTERN.fullmatch(requested)
    if requested.startswith("cuda"):
        if cuda_match is None:
            raise ValueError(
                f"Invalid CUDA device {device!r}; expected 'cuda' or 'cuda:<non-negative index>'"
            )
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")

        index = int(cuda_match.group(1) or 0)
        visible_devices = int(torch.cuda.device_count())
        if index >= visible_devices:
            raise RuntimeError(
                f"CUDA device index {index} is out of range; "
                f"{visible_devices} visible device(s)."
            )
        return DeviceInfo(name=requested, torch_device=torch.device(requested))

    if requested == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError("MPS was requested but torch.backends.mps.is_available() is False.")
        return DeviceInfo(name="mps", torch_device=torch.device("mps"))

    raise ValueError(f"Unsupported device: {device!r}")
