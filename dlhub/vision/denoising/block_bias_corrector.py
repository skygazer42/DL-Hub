from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def _pad_to_multiple_hw(x: torch.Tensor, multiple: int, *, mode: str = "replicate") -> tuple[torch.Tensor, tuple[int, int]]:
    if x.ndim != 4:
        raise ValueError(f"Expected NCHW, got {tuple(x.shape)}")
    m = int(multiple)
    if m <= 0:
        raise ValueError("multiple must be > 0")
    h, w = x.shape[-2:]
    pad_h = (m - (h % m)) % m
    pad_w = (m - (w % m)) % m
    if pad_h == 0 and pad_w == 0:
        return x, (0, 0)
    y = F.pad(x, (0, pad_w, 0, pad_h), mode=str(mode))
    return y, (int(pad_h), int(pad_w))


def _unpad_hw(x: torch.Tensor, pad_hw: tuple[int, int]) -> torch.Tensor:
    pad_h, pad_w = (int(pad_hw[0]), int(pad_hw[1]))
    if pad_h == 0 and pad_w == 0:
        return x
    h_end = None if pad_h == 0 else -pad_h
    w_end = None if pad_w == 0 else -pad_w
    return x[..., :h_end, :w_end]


class BlockBiasCorrector(nn.Module):
    """Remove block-wise additive bias by estimating per-block mean offsets."""

    def __init__(
        self,
        *,
        block_size: int = 8,
        strength: float = 1.0,
        padding: str = "replicate",
        clamp: bool = True,
    ) -> None:
        super().__init__()
        bs = int(block_size)
        if bs <= 0:
            raise ValueError("block_size must be > 0")
        s = float(strength)
        if s < 0.0:
            raise ValueError("strength must be >= 0")
        self.block_size = bs
        self.strength = s
        self.padding = str(padding)
        self.clamp = bool(clamp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        if x.ndim != 4:
            raise ValueError(f"Expected input shape (B, C, H, W), got {tuple(x.shape)}")

        if float(self.strength) == 0.0 or int(self.block_size) == 1:
            return x.clamp(0.0, 1.0) if self.clamp else x

        bs = int(self.block_size)
        x_pad, pad_hw = _pad_to_multiple_hw(x, bs, mode=self.padding)
        g = x_pad.mean(dim=(-2, -1), keepdim=True)

        blocks = F.avg_pool2d(x_pad, kernel_size=bs, stride=bs)  # (B,C,HB,WB)
        up = blocks.repeat_interleave(bs, dim=-2).repeat_interleave(bs, dim=-1)
        up = up[..., : x_pad.shape[-2], : x_pad.shape[-1]]

        bias = up - g
        y = x_pad - float(self.strength) * bias
        y = _unpad_hw(y, pad_hw)
        return y.clamp(0.0, 1.0) if self.clamp else y


_VARIANTS: dict[str, dict] = {
    "block_bias_tiny": {"block": 8},
    "block_bias_small": {"block": 6},
    "block_bias_base": {"block": 4},
}


def build_block_bias_corrector_denoiser(
    *,
    in_channels: int,  # unused
    sigma: float = 0.1,
    variant: str = "block_bias_tiny",
) -> nn.Module:
    _ = int(in_channels)
    _ = float(sigma)
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown BlockBiasCorrector variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    return BlockBiasCorrector(block_size=int(spec["block"]), strength=1.0, padding="replicate", clamp=True)


if __name__ == "__main__":
    torch.manual_seed(0)
    clean = torch.zeros(1, 1, 32, 32)
    clean[:, :, 10:22, 12:20] = 1.0
    # Add block bias (8x8).
    bias = torch.randn(1, 1, 4, 4) * 0.05
    bias = bias.repeat_interleave(8, dim=-2).repeat_interleave(8, dim=-1)
    noisy = (clean + bias).clamp(0.0, 1.0)
    m = build_block_bias_corrector_denoiser(in_channels=1, variant="block_bias_tiny")
    out = m(noisy)
    print("block_bias_tiny", tuple(out.shape), float((out - clean).pow(2).mean().item()))

