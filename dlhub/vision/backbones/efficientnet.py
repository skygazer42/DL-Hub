from __future__ import annotations

import math

import torch
from torch import nn

from dlhub.vision.backbones._blocks import ConvBNAct, DropPath, GlobalAvgPoolHead, SqueezeExcite, scale_channels


def _round_repeats(repeats: int, depth_mult: float) -> int:
    r = int(repeats)
    return max(1, int(math.ceil(r * float(depth_mult))))


class MBConv(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        *,
        stride: int,
        kernel_size: int,
        expand_ratio: float,
        se_ratio: float,
        dropout: float,
        drop_path: float,
    ) -> None:
        super().__init__()
        c_in = int(in_ch)
        c_out = int(out_ch)
        s = int(stride)
        k = int(kernel_size)
        if s not in {1, 2}:
            raise ValueError("stride must be 1 or 2")

        hidden = int(round(c_in * float(expand_ratio)))
        self.use_res = s == 1 and c_in == c_out

        layers: list[nn.Module] = []
        if hidden != c_in:
            layers.append(ConvBNAct(c_in, hidden, kernel_size=1, stride=1, padding=0, act="silu"))
        layers.append(
            ConvBNAct(
                hidden,
                hidden,
                kernel_size=k,
                stride=s,
                groups=hidden,
                act="silu",
            )
        )
        layers.append(SqueezeExcite(hidden, se_ratio=float(se_ratio)))
        layers.append(nn.Conv2d(hidden, c_out, kernel_size=1, bias=False))
        layers.append(nn.BatchNorm2d(c_out))
        self.net = nn.Sequential(*layers)
        self.drop = nn.Dropout2d(p=float(dropout)) if float(dropout) > 0 else nn.Identity()
        self.drop_path = DropPath(float(drop_path))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        y = self.net(x)
        y = self.drop(y)
        if self.use_res:
            y = self.drop_path(y)
            return x + y
        return y


class EfficientNetClassifier(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        width_mult: float,
        depth_mult: float,
        dropout: float,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        w = float(width_mult)
        d = float(depth_mult)

        stem_ch = scale_channels(32, w, min_ch=16, divisor=8)
        self.stem = ConvBNAct(int(in_channels), stem_ch, kernel_size=3, stride=2, act="silu")

        # EfficientNet-B0 base config: (exp, out, reps, stride, k)
        base: list[tuple[float, int, int, int, int]] = [
            (1.0, 16, 1, 1, 3),
            (6.0, 24, 2, 2, 3),
            (6.0, 40, 2, 2, 5),
            (6.0, 80, 3, 2, 3),
            (6.0, 112, 3, 1, 5),
            (6.0, 192, 4, 2, 5),
            (6.0, 320, 1, 1, 3),
        ]

        stages: list[nn.Module] = []
        in_ch = stem_ch
        total_blocks = sum(_round_repeats(r, d) for _, _, r, _, _ in base)
        block_idx = 0
        for exp, out_base, reps, stride, k in base:
            out_ch = scale_channels(int(out_base), w, min_ch=16, divisor=8)
            n = _round_repeats(int(reps), d)
            blocks: list[nn.Module] = []
            for i in range(n):
                s = int(stride) if i == 0 else 1
                dp = float(drop_path) * float(block_idx) / max(1, total_blocks - 1)
                blocks.append(
                    MBConv(
                        in_ch,
                        out_ch,
                        stride=s,
                        kernel_size=int(k),
                        expand_ratio=float(exp),
                        se_ratio=0.25,
                        dropout=float(dropout),
                        drop_path=dp,
                    )
                )
                in_ch = out_ch
                block_idx += 1
            stages.append(nn.Sequential(*blocks))
        self.stages = nn.Sequential(*stages)

        head_ch = scale_channels(1280, w, min_ch=128, divisor=8)
        self.head_conv = ConvBNAct(in_ch, head_ch, kernel_size=1, stride=1, padding=0, act="silu")
        self.head = GlobalAvgPoolHead(head_ch, int(num_classes), dropout=float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        x = self.stem(x)
        x = self.stages(x)
        x = self.head_conv(x)
        return self.head(x)


_VARIANTS: dict[str, tuple[float, float]] = {
    # (width_mult, depth_mult) – standard EfficientNet scaling.
    "b0": (1.0, 1.0),
    "b1": (1.0, 1.1),
    "b2": (1.1, 1.2),
    "b3": (1.2, 1.4),
    "b4": (1.4, 1.8),
}


def build_efficientnet_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "b0",
    width_mult: float = 1.0,
    dropout: float | None = None,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name.startswith("efficientnet_"):
        name = name.split("_", 1)[1]
    if name not in _VARIANTS:
        raise ValueError("Unknown EfficientNet variant. Supported: b0|b1|b2|b3|b4")
    w0, d0 = _VARIANTS[name]
    if dropout is None:
        dropout = 0.1
    return EfficientNetClassifier(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        width_mult=float(width_mult) * float(w0),
        depth_mult=float(d0),
        dropout=float(dropout),
        drop_path=0.1,
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    for v in ["b0", "b1", "b4"]:
        m = build_efficientnet_classifier(in_channels=3, num_classes=10, variant=v, width_mult=0.75)
        y = m(x)
        print(f"efficientnet_{v}", tuple(y.shape))

