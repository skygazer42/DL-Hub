from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from dlhub.vision.backbones._blocks import ConvBNAct
from dlhub.vision.segmentation._common import check_nchw


class _EncBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, *, depth: int) -> None:
        super().__init__()
        c_in = int(in_ch)
        c_out = int(out_ch)
        d = int(depth)
        if d <= 0:
            raise ValueError("depth must be > 0")
        layers: list[nn.Module] = [ConvBNAct(c_in, c_out, kernel_size=3, stride=1, act="relu")]
        for _ in range(d - 1):
            layers.append(ConvBNAct(c_out, c_out, kernel_size=3, stride=1, act="relu"))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SegNet(nn.Module):
    """SegNet semantic segmentation (toy-first).

    Uses an encoder-decoder with max-pooling and upsampling.
    (This compact implementation uses nearest upsampling instead of pooling indices.)
    """

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        base_channels: int = 32,
        levels: int = 4,
        depth: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        nc = int(num_classes)
        if nc <= 0:
            raise ValueError("num_classes must be > 0")
        base = int(base_channels)
        lv = int(levels)
        d = int(depth)
        if base < 8:
            raise ValueError("base_channels must be >= 8")
        if lv < 2:
            raise ValueError("levels must be >= 2")
        if d <= 0:
            raise ValueError("depth must be > 0")

        enc: list[nn.Module] = []
        ch = int(in_channels)
        cur = base
        for _ in range(lv):
            enc.append(_EncBlock(ch, cur, depth=d))
            ch = cur
            cur *= 2
        self.enc = nn.ModuleList(enc)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        dec: list[nn.Module] = []
        cur = ch
        for _ in range(lv - 1, -1, -1):
            nxt = max(base, cur // 2)
            dec.append(_EncBlock(cur, nxt, depth=d))
            cur = nxt
        self.dec = nn.ModuleList(dec)

        self.drop = nn.Dropout2d(p=float(dropout))
        self.out = nn.Conv2d(base, nc, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = check_nchw(x)
        inp_hw = x.shape[-2:]

        feats: list[torch.Tensor] = []
        for block in self.enc:
            x = block(x)
            feats.append(x)
            x = self.pool(x)

        for block in self.dec:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
            x = block(x)

        x = self.drop(x)
        logits = self.out(x)
        return F.interpolate(logits, size=inp_hw, mode="nearest")


_VARIANTS: dict[str, dict] = {
    "segnet_tiny": {"base_channels": 16, "levels": 3, "depth": 1, "dropout": 0.0},
    "segnet_small": {"base_channels": 24, "levels": 4, "depth": 2, "dropout": 0.0},
    "segnet_base": {"base_channels": 32, "levels": 4, "depth": 2, "dropout": 0.1},
}


def build_segnet_segmenter(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "segnet_small",
    dropout: float | None = None,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown SegNet variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    p = float(spec["dropout"]) if dropout is None else float(dropout)
    return SegNet(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        base_channels=int(spec["base_channels"]),
        levels=int(spec["levels"]),
        depth=int(spec["depth"]),
        dropout=float(p),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_segnet_segmenter(in_channels=3, num_classes=4, variant="segnet_tiny")
    y = m(x)
    print("segnet_tiny", tuple(y.shape))
    loss = y.mean()
    loss.backward()
    print("ok")

