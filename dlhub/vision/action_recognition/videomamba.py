"""VideoMamba (Mamba-style sequence mixing) - toy-first video action classifier.

Reference:
- "VideoMamba: State Space Model for Efficient Video Understanding" (arXiv 2024)

Toy interpretation:
- Tubelet embedding -> token sequence (no pretrained weights).
- Replace attention with a light "Mamba-inspired" sequence mixer:
  depthwise Conv1d over tokens + GLU-style gating.
"""

import torch
from torch import nn

from dlhub.vision.backbones._blocks import DropPath, scale_channels

from ._common import check_video_input


class VideoMambaBlock(nn.Module):
    """A tiny Mamba-like token mixer (attention-free)."""

    def __init__(self, dim: int, *, kernel_size: int = 7, drop_path: float = 0.0) -> None:
        super().__init__()
        d = int(dim)
        k = int(kernel_size)
        if d <= 0:
            raise ValueError("dim must be > 0")
        if k <= 0 or k % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")

        self.norm = nn.LayerNorm(d)
        self.dw = nn.Conv1d(d, d, kernel_size=k, padding=k // 2, groups=d, bias=False)
        self.proj = nn.Linear(d, 2 * d, bias=True)
        self.out = nn.Linear(d, d, bias=True)
        self.dp = DropPath(float(drop_path))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.norm(x)
        y = self.dw(x.transpose(1, 2)).transpose(1, 2)
        u, v = self.proj(y).chunk(2, dim=-1)
        y = u * torch.sigmoid(v)
        y = self.out(y)
        return identity + self.dp(y)


class VideoMambaVideoClassifier(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        image_size: int,
        frames: int,
        patch_size: int,
        tubelet: int,
        dim: int,
        depth: int,
        drop_path: float,
        dropout: float,
    ) -> None:
        super().__init__()
        img = int(image_size)
        t = int(frames)
        p = int(patch_size)
        tb = int(tubelet)
        d = int(dim)
        n = int(depth)
        if img <= 0 or t <= 0:
            raise ValueError("image_size and frames must be > 0")
        if p <= 0 or img % p != 0:
            raise ValueError("patch_size must be > 0 and divide image_size")
        if tb <= 0 or t % tb != 0:
            raise ValueError("tubelet must be > 0 and divide frames")
        if d <= 0 or n <= 0:
            raise ValueError("dim/depth must be > 0")

        self.frames = t
        self.image_size = img

        t_grid = t // tb
        h_grid = img // p
        w_grid = img // p
        num_tokens = int(t_grid * h_grid * w_grid)
        self.num_tokens = num_tokens

        self.tubelet_embed = nn.Conv3d(
            int(in_channels),
            d,
            kernel_size=(tb, p, p),
            stride=(tb, p, p),
            padding=0,
            bias=True,
        )
        self.pos = nn.Parameter(torch.zeros(1, num_tokens, d))

        dp_rates = torch.linspace(0.0, float(drop_path), steps=n).tolist()
        self.blocks = nn.Sequential(
            *[VideoMambaBlock(d, kernel_size=7, drop_path=float(dp_rates[i])) for i in range(n)]
        )
        self.norm = nn.LayerNorm(d)
        self.drop = nn.Dropout(p=float(dropout))
        self.head = nn.Linear(d, int(num_classes))

        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = check_video_input(x)
        b, c, t, h, w = x.shape
        if int(t) != int(self.frames):
            raise ValueError(f"Expected T={self.frames} frames for this model, got T={t}")
        if int(h) != int(self.image_size) or int(w) != int(self.image_size):
            raise ValueError(f"Expected H=W={self.image_size} for this model, got (H,W)=({h},{w})")

        tok = self.tubelet_embed(x).flatten(2).transpose(1, 2).contiguous()  # (B, N, D)
        if int(tok.shape[1]) != int(self.num_tokens):
            raise ValueError(
                f"Unexpected token count: got N={tok.shape[1]}, expected {self.num_tokens}"
            )
        tok = tok + self.pos
        tok = self.blocks(tok)
        tok = self.norm(tok)
        pooled = tok.mean(dim=1)
        pooled = self.drop(pooled)
        return self.head(pooled)


_VARIANTS: dict[str, dict] = {
    "videomamba_tiny": {"patch": 8, "tubelet": 2, "dim": 160, "depth": 4},
    "videomamba_small": {"patch": 8, "tubelet": 2, "dim": 192, "depth": 6},
    "videomamba_base": {"patch": 4, "tubelet": 2, "dim": 256, "depth": 8},
}


def build_videomamba_video_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "videomamba_small",
    image_size: int = 64,
    frames: int = 8,
    width_mult: float = 1.0,
    dropout: float = 0.1,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown VideoMamba variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    dim = scale_channels(int(spec["dim"]), float(width_mult), min_ch=32, divisor=8)
    return VideoMambaVideoClassifier(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        image_size=int(image_size),
        frames=int(frames),
        patch_size=int(spec["patch"]),
        tubelet=int(spec["tubelet"]),
        dim=int(dim),
        depth=int(spec["depth"]),
        drop_path=0.1,
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 8, 64, 64)
    m = build_videomamba_video_classifier(
        in_channels=3,
        num_classes=6,
        variant="videomamba_tiny",
        image_size=64,
        frames=8,
        width_mult=0.5,
        dropout=0.0,
    )
    y = m(x)
    print("videomamba_tiny", tuple(y.shape))
    loss = y.mean()
    loss.backward()
    print("ok")
