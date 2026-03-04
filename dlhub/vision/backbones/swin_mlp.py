from __future__ import annotations

import torch
from torch import nn

from dlhub.vision.backbones._blocks import DropPath
from dlhub.vision.backbones._transformer import MLP, PatchEmbed


def _window_partition(x: torch.Tensor, window: int) -> torch.Tensor:
    b, h, w, c = x.shape
    ws = int(window)
    x = x.view(b, h // ws, ws, w // ws, ws, c)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b * (h // ws) * (w // ws), ws * ws, c)
    return x


def _window_reverse(windows: torch.Tensor, window: int, h: int, w: int) -> torch.Tensor:
    ws = int(window)
    b = int(windows.shape[0] // (h // ws * w // ws))
    x = windows.view(b, h // ws, w // ws, ws, ws, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)
    return x


class SwinMLPTokenMix(nn.Module):
    """Window token-mixing MLP (mix along token dimension inside each window)."""

    def __init__(self, window: int) -> None:
        super().__init__()
        ws = int(window)
        self.ws = ws
        self.mlp = nn.Linear(ws * ws, ws * ws)

    def forward(self, x: torch.Tensor, *, hw: tuple[int, int]) -> torch.Tensor:
        b, n, c = x.shape
        h, w = int(hw[0]), int(hw[1])
        ws = self.ws
        x2d = x.view(b, h, w, c)
        win = _window_partition(x2d, ws)  # (B*nw, ws*ws, C)
        win = win.transpose(1, 2)  # (B*nw, C, ws*ws)
        win = self.mlp(win)
        win = win.transpose(1, 2)
        x2d = _window_reverse(win, ws, h, w)
        return x2d.view(b, n, c)


class SwinMLPBlock(nn.Module):
    def __init__(self, dim: int, *, window: int = 8, drop_path: float = 0.0) -> None:
        super().__init__()
        d = int(dim)
        self.window = int(window)
        self.norm1 = nn.LayerNorm(d)
        self.mix = SwinMLPTokenMix(self.window)
        self.dp1 = DropPath(float(drop_path))
        self.norm2 = nn.LayerNorm(d)
        self.mlp = MLP(d, 4 * d, dropout=0.0, act="gelu")
        self.dp2 = DropPath(float(drop_path))

    def forward(self, x: torch.Tensor, *, hw: tuple[int, int]) -> torch.Tensor:
        x = x + self.dp1(self.mix(self.norm1(x), hw=hw))
        x = x + self.dp2(self.mlp(self.norm2(x)))
        return x


class SwinMLPClassifier(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        image_size: int = 64,
        patch_size: int = 4,
        dim: int = 192,
        depth: int = 8,
        window: int = 8,
        drop_path: float = 0.1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.patch = PatchEmbed(int(in_channels), int(dim), patch_size=int(patch_size))
        h = int(image_size) // int(patch_size)
        w = int(image_size) // int(patch_size)
        self.hw = (h, w)
        self.pos = nn.Parameter(torch.zeros(1, h * w, int(dim)))
        dp_rates = torch.linspace(0.0, float(drop_path), steps=int(depth)).tolist()
        self.blocks = nn.ModuleList([SwinMLPBlock(int(dim), window=int(window), drop_path=float(dp_rates[i])) for i in range(int(depth))])
        self.norm = nn.LayerNorm(int(dim))
        self.drop = nn.Dropout(p=float(dropout))
        self.head = nn.Linear(int(dim), int(num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        x = self.patch(x) + self.pos
        for b in self.blocks:
            x = b(x, hw=self.hw)
        x = self.norm(x)
        x = self.drop(x.mean(dim=1))
        return self.head(x)


_VARIANTS: dict[str, dict] = {
    "swin_mlp_tiny": {"dim": 192, "depth": 8, "patch": 4, "window": 8},
    "swin_mlp_small": {"dim": 256, "depth": 10, "patch": 4, "window": 8},
}


def build_swin_mlp_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "swin_mlp_tiny",
    image_size: int = 64,
    drop_path: float = 0.1,
    dropout: float = 0.1,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown SwinMLP variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    return SwinMLPClassifier(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        image_size=int(image_size),
        patch_size=int(spec["patch"]),
        dim=int(spec["dim"]),
        depth=int(spec["depth"]),
        window=int(spec["window"]),
        drop_path=float(drop_path),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_swin_mlp_classifier(in_channels=3, num_classes=10, variant="swin_mlp_tiny", image_size=64)
    y = m(x)
    print("swin_mlp_tiny", tuple(y.shape))

