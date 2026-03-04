from __future__ import annotations

import math

import torch
from torch import nn

from dlhub.vision.backbones._blocks import GlobalAvgPoolHead


class HaarScattering2D(nn.Module):
    """A tiny Haar wavelet scattering front-end (fixed filters).

    This is not a full scattering transform implementation, but it captures the
    core idea: fixed multi-scale, multi-orientation filter banks.
    """

    def __init__(self, in_channels: int, *, levels: int = 2) -> None:
        super().__init__()
        c = int(in_channels)
        l = int(levels)
        if c <= 0:
            raise ValueError("in_channels must be > 0")
        if l <= 0:
            raise ValueError("levels must be > 0")
        self.levels = l

        # 2x2 Haar filters: LL, LH, HL, HH
        ll = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
        lh = torch.tensor([[-1.0, -1.0], [1.0, 1.0]])
        hl = torch.tensor([[-1.0, 1.0], [-1.0, 1.0]])
        hh = torch.tensor([[1.0, -1.0], [-1.0, 1.0]])
        bank = torch.stack([ll, lh, hl, hh], dim=0) / 2.0  # (4, 2, 2)
        weight = bank[:, None, :, :].repeat(c, 1, 1, 1)  # (4*C, 1, 2, 2)
        self.register_buffer("weight", weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        out = x
        feats: list[torch.Tensor] = []
        for _ in range(self.levels):
            # group depthwise conv: each channel gets 4 outputs
            y = nn.functional.conv2d(out, self.weight, stride=2, padding=0, groups=out.shape[1])
            feats.append(y)
            # Keep LL part as next input (first of the 4 per channel)
            b, c4, h, w = y.shape
            c = c4 // 4
            out = y.view(b, c, 4, h, w)[:, :, 0]
        return torch.cat(feats, dim=1)


class ScatNetClassifier(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        levels: int = 2,
        hidden: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.scat = HaarScattering2D(int(in_channels), levels=int(levels))
        # Each level outputs 4*C channels, total levels*(4*C) channels.
        feat_ch = int(levels) * 4 * int(in_channels)
        self.proj = nn.Sequential(
            nn.Conv2d(feat_ch, int(hidden), kernel_size=1, bias=False),
            nn.BatchNorm2d(int(hidden)),
            nn.ReLU(inplace=True),
        )
        self.head = GlobalAvgPoolHead(int(hidden), int(num_classes), dropout=float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.scat(x)
        x = self.proj(x)
        return self.head(x)


_VARIANTS: dict[str, dict] = {
    "scatternet_l1": {"levels": 1, "hidden": 192},
    "scatternet_l2": {"levels": 2, "hidden": 256},
    "scatternet_l3": {"levels": 3, "hidden": 320},
}


def build_scatternet_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "scatternet_l2",
    dropout: float = 0.1,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown ScatNet variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    return ScatNetClassifier(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        levels=int(spec["levels"]),
        hidden=int(spec["hidden"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    for v in ["scatternet_l1", "scatternet_l2"]:
        m = build_scatternet_classifier(in_channels=3, num_classes=10, variant=v)
        y = m(x)
        print(v, tuple(y.shape))

