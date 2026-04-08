from __future__ import annotations

import torch
from torch import nn

from dlhub.vision.denoising._utils import pad_to_multiple, unpad

from ._common import PixelShuffleUpsampler, check_low_res_image, validate_upscale_factor


def _window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    b, h, w, c = x.shape
    ws = int(window_size)
    x = x.view(b, h // ws, ws, w // ws, ws, c)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return x.view(b * (h // ws) * (w // ws), ws * ws, c)


def _window_reverse(
    windows: torch.Tensor, window_size: int, *, batch_size: int, height: int, width: int, channels: int
) -> torch.Tensor:
    ws = int(window_size)
    x = windows.view(batch_size, height // ws, width // ws, ws, ws, channels)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return x.view(batch_size, height, width, channels)


class WindowBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        window_size: int,
        shift: bool,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        d = int(dim)
        heads = int(num_heads)
        ws = int(window_size)
        if d <= 0:
            raise ValueError("dim must be > 0")
        if heads <= 0 or d % heads != 0:
            raise ValueError("num_heads must be > 0 and divide dim")
        if ws <= 0:
            raise ValueError("window_size must be > 0")
        hidden = max(d, int(round(d * float(mlp_ratio))))

        self.window_size = ws
        self.shift_size = ws // 2 if bool(shift) else 0
        self.norm1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(
            embed_dim=d,
            num_heads=heads,
            dropout=float(dropout),
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(
            nn.Linear(d, hidden),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden, d),
            nn.Dropout(float(dropout)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, h, w, c = x.shape
        ws = self.window_size
        if h % ws != 0 or w % ws != 0:
            raise ValueError("Input must be padded so H and W are divisible by window_size")

        shortcut = x
        if self.shift_size:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

        windows = _window_partition(x, ws)
        normed = self.norm1(windows)
        attended, _ = self.attn(normed, normed, normed, need_weights=False)
        x = _window_reverse(attended, ws, batch_size=b, height=h, width=w, channels=c)

        if self.shift_size:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x


class FocusSR(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        c_in = int(in_channels)
        embed = int(embed_dim)
        num_blocks = int(depth)
        heads = int(num_heads)
        ws = int(window_size)
        if c_in <= 0:
            raise ValueError("in_channels must be > 0")
        if embed <= 0:
            raise ValueError("embed_dim must be > 0")
        if num_blocks <= 0:
            raise ValueError("depth must be > 0")
        if heads <= 0 or embed % heads != 0:
            raise ValueError("num_heads must be > 0 and divide embed_dim")

        self.window_size = ws
        self.head = nn.Conv2d(c_in, embed, kernel_size=3, padding=1, bias=True)
        self.blocks = nn.ModuleList(
            [
                WindowBlock(
                    embed,
                    heads,
                    window_size=ws,
                    shift=bool(i % 2 == 1),
                    mlp_ratio=float(mlp_ratio),
                    dropout=float(dropout),
                )
                for i in range(num_blocks)
            ]
        )
        self.body_tail = nn.Conv2d(embed, embed, kernel_size=3, padding=1, bias=True)
        self.upsample = PixelShuffleUpsampler(embed, upscale_factor=2)
        self.tail = nn.Conv2d(embed, c_in, kernel_size=3, padding=1, bias=True)

    def forward(self, low_res: torch.Tensor) -> dict[str, torch.Tensor]:
        x = check_low_res_image(low_res)
        x, pad_hw = pad_to_multiple(x, self.window_size, mode="reflect")
        feat = self.head(x)
        y = feat.permute(0, 2, 3, 1).contiguous()
        for block in self.blocks:
            y = block(y)
        y = y.permute(0, 3, 1, 2).contiguous()
        y = self.body_tail(y) + feat
        y = self.upsample(y)
        sr = self.tail(y)
        sr = unpad(sr, (int(pad_hw[0]) * 2, int(pad_hw[1]) * 2))
        return {"sr": sr}


_VARIANTS: dict[str, dict[str, float | int]] = {
    "focus_sr_tiny": {"embed": 24, "depth": 2, "heads": 4, "window": 4, "mlp": 2.0},
    "focus_sr_small": {"embed": 32, "depth": 3, "heads": 4, "window": 4, "mlp": 2.0},
    "focus_sr_base": {"embed": 48, "depth": 4, "heads": 6, "window": 4, "mlp": 2.0},
}


def build_focus_sr_super_resolver(
    *,
    in_channels: int,
    variant: str = "focus_sr_small",
    upscale_factor: int = 2,
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    validate_upscale_factor(upscale_factor)

    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(
            f"Unknown SwinIR-SR variant: {variant!r}. Supported: {sorted(_VARIANTS)}"
        )
    spec = _VARIANTS[name]
    embed = max(8, int(int(spec["embed"]) * float(width_mult)))
    heads = max(1, int(spec["heads"]))
    while embed % heads != 0 and heads > 1:
        heads -= 1
    return FocusSR(
        in_channels=int(in_channels),
        embed_dim=embed,
        depth=int(spec["depth"]),
        num_heads=heads,
        window_size=int(spec["window"]),
        mlp_ratio=float(spec["mlp"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    m = build_focus_sr_super_resolver(in_channels=3, variant="focus_sr_tiny")
    x = torch.randn(2, 3, 16, 16)
    y = m(x)
    print("focus_sr_tiny", tuple(y["sr"].shape))

