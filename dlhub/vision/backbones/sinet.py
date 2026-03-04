from __future__ import annotations

import torch
from torch import nn

from dlhub.vision.backbones._blocks import ConvBNAct, DropPath, GlobalAvgPoolHead, SqueezeExcite, scale_channels


class SiBlock(nn.Module):
    """SINet-ish selective fusion block (simplified)."""

    def __init__(self, in_ch: int, out_ch: int, *, stride: int = 1, drop_path: float = 0.0) -> None:
        super().__init__()
        c_in = int(in_ch)
        c_out = int(out_ch)
        s = int(stride)
        self.local = ConvBNAct(c_in, c_out, kernel_size=3, stride=s, act="relu")
        self.proj = ConvBNAct(c_in, c_out, kernel_size=1, stride=s, padding=0, act="relu")
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(c_out, c_out, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.se = SqueezeExcite(c_out, se_ratio=0.25)
        self.dp = DropPath(float(drop_path))
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.local(x)
        g = self.gate(y)
        y = y * g + self.proj(x)
        y = self.se(y)
        y = self.dp(y)
        return self.act(y)


class SINetClassifier(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        dims: tuple[int, int, int, int],
        depths: tuple[int, int, int, int],
        width_mult: float = 1.0,
        drop_path: float = 0.1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        dims = tuple(scale_channels(int(d), float(width_mult), min_ch=16, divisor=8) for d in dims)
        depths = tuple(int(d) for d in depths)

        total = sum(depths)
        dp_rates = torch.linspace(0.0, float(drop_path), steps=max(1, total)).tolist()
        dp_iter = iter(dp_rates)

        self.stem = nn.Sequential(
            ConvBNAct(int(in_channels), dims[0], kernel_size=3, stride=2, act="relu"),
            ConvBNAct(dims[0], dims[0], kernel_size=3, stride=1, act="relu"),
        )

        def stage(in_ch: int, out_ch: int, depth: int, *, stride: int) -> nn.Sequential:
            blocks: list[nn.Module] = [SiBlock(in_ch, out_ch, stride=int(stride), drop_path=float(next(dp_iter, 0.0)))]
            for _ in range(int(depth) - 1):
                blocks.append(SiBlock(out_ch, out_ch, stride=1, drop_path=float(next(dp_iter, 0.0))))
            return nn.Sequential(*blocks)

        self.stage1 = stage(dims[0], dims[0], depths[0], stride=1)
        self.stage2 = stage(dims[0], dims[1], depths[1], stride=2)
        self.stage3 = stage(dims[1], dims[2], depths[2], stride=2)
        self.stage4 = stage(dims[2], dims[3], depths[3], stride=2)

        self.head = GlobalAvgPoolHead(dims[3], int(num_classes), dropout=float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return self.head(x)


_VARIANTS: dict[str, dict] = {
    "sinet_tiny": {"dims": (48, 96, 192, 384), "depths": (1, 2, 4, 2)},
    "sinet_small": {"dims": (64, 128, 256, 512), "depths": (2, 2, 6, 2)},
    "sinet_base": {"dims": (80, 160, 320, 640), "depths": (2, 3, 8, 3)},
}


def build_sinet_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "sinet_tiny",
    width_mult: float = 1.0,
    drop_path: float = 0.1,
    dropout: float = 0.1,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown SINet variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    return SINetClassifier(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        dims=tuple(map(int, spec["dims"])),
        depths=tuple(map(int, spec["depths"])),
        width_mult=float(width_mult),
        drop_path=float(drop_path),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_sinet_classifier(in_channels=3, num_classes=10, variant="sinet_tiny", width_mult=0.5)
    y = m(x)
    print("sinet_tiny", tuple(y.shape))
