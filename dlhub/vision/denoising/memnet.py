import torch
import torch.nn.functional as F
from torch import nn


class _ResBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        c = int(channels)
        self.conv1 = nn.Conv2d(c, c, kernel_size=3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(c, c, kernel_size=3, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.conv1(x), inplace=True)
        y = self.conv2(y)
        return F.relu(x + y, inplace=True)


class _MemoryBlock(nn.Module):
    """MemNet-style memory block with short-term + long-term feature fusion."""

    def __init__(self, channels: int, *, num_resblocks: int, num_prev_mem: int) -> None:
        super().__init__()
        c = int(channels)
        r = int(num_resblocks)
        p = int(num_prev_mem)
        if c <= 0:
            raise ValueError("channels must be > 0")
        if r <= 0:
            raise ValueError("num_resblocks must be > 0")
        if p < 0:
            raise ValueError("num_prev_mem must be >= 0")

        self.num_prev_mem = p
        self.resblocks = nn.ModuleList([_ResBlock(c) for _ in range(r)])

        # Long-term memories: p tensors, each with c channels.
        # Short-term memories: input + r intermediate outputs => (r + 1) tensors.
        mem_ch = c * (p + (r + 1))
        self.gate = nn.Conv2d(mem_ch, c, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor, long_term: list[torch.Tensor]) -> torch.Tensor:
        if len(long_term) != self.num_prev_mem:
            # The expected count is fixed per block so the 1x1 conv has static input channels.
            raise ValueError(
                f"MemNet internal error: expected {self.num_prev_mem} long-term memories, got {len(long_term)}"
            )

        stms: list[torch.Tensor] = [x]
        y = x
        for rb in self.resblocks:
            y = rb(y)
            stms.append(y)

        mem = torch.cat([*long_term, *stms], dim=1)
        out = self.gate(mem)
        return F.relu(out, inplace=True)


class MemNet(nn.Module):
    """MemNet (Persistent Memory Network) adapted for denoising (compact-first, pure torch).

    MemNet is typically used for image restoration; here we keep resolution and predict a residual/noise map.
    Output is `x - residual`.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        features: int = 64,
        num_memblocks: int = 6,
        num_resblocks: int = 6,
    ) -> None:
        super().__init__()
        c_in = int(in_channels)
        f = int(features)
        m = int(num_memblocks)
        r = int(num_resblocks)
        if c_in <= 0:
            raise ValueError("in_channels must be > 0")
        if f < 8:
            raise ValueError("features must be >= 8")
        if m <= 0:
            raise ValueError("num_memblocks must be > 0")
        if r <= 0:
            raise ValueError("num_resblocks must be > 0")

        self.in_conv = nn.Conv2d(c_in, f, kernel_size=3, padding=1, bias=True)

        blocks: list[_MemoryBlock] = []
        for i in range(m):
            blocks.append(_MemoryBlock(f, num_resblocks=r, num_prev_mem=i))
        self.blocks = nn.ModuleList(blocks)

        self.out_conv = nn.Conv2d(f, c_in, kernel_size=3, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        if x.ndim != 4:
            raise ValueError(f"Expected input shape (B, C, H, W), got {tuple(x.shape)}")

        y = F.relu(self.in_conv(x), inplace=True)
        long_term: list[torch.Tensor] = []
        for blk in self.blocks:
            y = blk(y, long_term)
            long_term.append(y)

        residual = self.out_conv(y)
        return x - residual


_VARIANTS: dict[str, dict] = {
    "memnet_tiny": {"features": 32, "num_memblocks": 3, "num_resblocks": 3},
    "memnet_small": {"features": 48, "num_memblocks": 6, "num_resblocks": 4},
    "memnet_base": {"features": 64, "num_memblocks": 8, "num_resblocks": 6},
}


def build_memnet_denoiser(*, in_channels: int, variant: str = "memnet_small") -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown MemNet variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    return MemNet(
        in_channels=int(in_channels),
        features=int(spec["features"]),
        num_memblocks=int(spec["num_memblocks"]),
        num_resblocks=int(spec["num_resblocks"]),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.rand(2, 1, 64, 64)
    noisy = (x + torch.randn_like(x) * 0.1).clamp(0.0, 1.0)
    m = build_memnet_denoiser(in_channels=1, variant="memnet_tiny")
    y = m(noisy)
    print("memnet_tiny", tuple(y.shape))
    loss = (y - x).pow(2).mean()
    loss.backward()
    print("ok")
