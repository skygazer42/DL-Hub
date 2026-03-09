
import torch
import torch.nn.functional as F


def pad_to_multiple(
    x: torch.Tensor,
    multiple: int,
    *,
    mode: str = "reflect",
) -> tuple[torch.Tensor, tuple[int, int]]:
    """Pad NCHW tensor on bottom/right so H and W are divisible by `multiple`."""

    m = int(multiple)
    if m <= 0:
        raise ValueError("multiple must be > 0")
    if x.ndim != 4:
        raise ValueError(f"Expected NCHW tensor, got shape={tuple(x.shape)}")

    h, w = x.shape[-2:]
    pad_h = (m - (h % m)) % m
    pad_w = (m - (w % m)) % m
    if pad_h == 0 and pad_w == 0:
        return x, (0, 0)
    y = F.pad(x, (0, pad_w, 0, pad_h), mode=str(mode))
    return y, (int(pad_h), int(pad_w))


def unpad(
    x: torch.Tensor,
    pad_hw: tuple[int, int],
) -> torch.Tensor:
    """Undo `pad_to_multiple`."""

    pad_h, pad_w = (int(pad_hw[0]), int(pad_hw[1]))
    if pad_h == 0 and pad_w == 0:
        return x
    h_end = None if pad_h == 0 else -pad_h
    w_end = None if pad_w == 0 else -pad_w
    return x[..., :h_end, :w_end]

