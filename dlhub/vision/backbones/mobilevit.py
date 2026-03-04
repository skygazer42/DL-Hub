from __future__ import annotations

import torch
from torch import nn

from dlhub.vision.backbones._blocks import ConvBNAct, DropPath, SqueezeExcite, scale_channels
from dlhub.vision.backbones._transformer import TransformerEncoderBlock


def _pick_heads(embed_dim: int, preferred: list[int]) -> int:
    d = int(embed_dim)
    for h in preferred:
        if h > 0 and d % int(h) == 0:
            return int(h)
    return 1


class MBConv(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        *,
        stride: int,
        expand_ratio: float = 2.0,
        se_ratio: float = 0.25,
        dropout: float = 0.0,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        c_in = int(in_ch)
        c_out = int(out_ch)
        s = int(stride)
        if s not in {1, 2}:
            raise ValueError("stride must be 1 or 2")

        hidden = int(round(c_in * float(expand_ratio)))
        self.use_res = s == 1 and c_in == c_out

        layers: list[nn.Module] = []
        if hidden != c_in:
            layers.append(ConvBNAct(c_in, hidden, kernel_size=1, stride=1, padding=0, act="silu"))
        layers.append(ConvBNAct(hidden, hidden, kernel_size=3, stride=s, groups=hidden, act="silu"))
        se = float(se_ratio)
        layers.append(SqueezeExcite(hidden, se_ratio=se) if se > 0 else nn.Identity())
        layers.append(nn.Conv2d(hidden, c_out, kernel_size=1, bias=False))
        layers.append(nn.BatchNorm2d(c_out))
        self.net = nn.Sequential(*layers)
        self.drop = nn.Dropout2d(p=float(dropout)) if float(dropout) > 0 else nn.Identity()
        self.drop_path = DropPath(float(drop_path))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        out = self.net(x)
        out = self.drop(out)
        if self.use_res:
            out = self.drop_path(out)
            return x + out
        return out


class MobileViTBlock(nn.Module):
    def __init__(
        self,
        *,
        in_ch: int,
        out_ch: int,
        transformer_dim: int,
        num_heads: int,
        depth: int,
        dropout: float,
    ) -> None:
        super().__init__()
        in_ch = int(in_ch)
        out_ch = int(out_ch)
        tdim = int(transformer_dim)
        heads = int(num_heads)
        depth = int(depth)
        if depth <= 0:
            raise ValueError("depth must be > 0")

        self.local = nn.Sequential(
            ConvBNAct(in_ch, in_ch, kernel_size=3, stride=1, groups=in_ch, act="silu"),
            ConvBNAct(in_ch, tdim, kernel_size=1, stride=1, padding=0, act="silu"),
        )
        self.transformer = nn.Sequential(
            *[TransformerEncoderBlock(dim=tdim, num_heads=heads, dropout=float(dropout)) for _ in range(depth)]
        )
        self.proj = ConvBNAct(tdim, out_ch, kernel_size=1, stride=1, padding=0, act="silu")
        self.fuse = ConvBNAct(in_ch + out_ch, out_ch, kernel_size=3, stride=1, act="silu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        b, _, h, w = x.shape
        y = self.local(x)  # (B, tdim, H, W)
        y = y.flatten(2).transpose(1, 2)  # (B, T, tdim)
        y = self.transformer(y)
        y = y.transpose(1, 2).contiguous().view(b, -1, h, w)
        y = self.proj(y)
        return self.fuse(torch.cat([x, y], dim=1))


class MobileViTClassifier(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        image_size: int = 64,
        variant: str = "mobilevit_tiny",
        width_mult: float = 1.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if int(image_size) % 16 != 0:
            raise ValueError("MobileViT expects image_size divisible by 16")

        name = str(variant).lower().strip()
        w = float(width_mult)

        if name in {"mobilevit_tiny", "tiny", "mobilevit"}:
            widths = (16, 24, 32, 64, 96)
            transformer_dims = (64, 96)
            transformer_depths = (2, 2)
        elif name in {"mobilevit_small", "small"}:
            widths = (16, 32, 48, 80, 128)
            transformer_dims = (96, 128)
            transformer_depths = (2, 4)
        elif name in {"mobilevit_base", "base"}:
            widths = (24, 48, 64, 128, 192)
            transformer_dims = (128, 192)
            transformer_depths = (4, 4)
        else:
            raise ValueError("Unknown MobileViT variant. Supported: mobilevit_tiny|mobilevit_small|mobilevit_base")

        w1, w2, w3, w4, w5 = (scale_channels(int(c), w, min_ch=16, divisor=8) for c in widths)
        t1, t2 = (scale_channels(int(c), w, min_ch=64, divisor=8) for c in transformer_dims)
        d1, d2 = (int(d) for d in transformer_depths)

        self.stem = ConvBNAct(int(in_channels), w1, kernel_size=3, stride=2, act="silu")
        self.stage1 = MBConv(w1, w2, stride=1, expand_ratio=2, se_ratio=0.0, dropout=float(dropout))
        self.stage2 = MBConv(w2, w3, stride=2, expand_ratio=2, se_ratio=0.25, dropout=float(dropout))
        self.stage3 = nn.Sequential(
            MBConv(w3, w4, stride=2, expand_ratio=2, se_ratio=0.25, dropout=float(dropout)),
            MobileViTBlock(
                in_ch=w4,
                out_ch=w4,
                transformer_dim=t1,
                num_heads=_pick_heads(t1, preferred=[8, 6, 4, 3, 2, 1]),
                depth=d1,
                dropout=float(dropout),
            ),
        )
        self.stage4 = nn.Sequential(
            MBConv(w4, w5, stride=2, expand_ratio=2, se_ratio=0.25, dropout=float(dropout)),
            MobileViTBlock(
                in_ch=w5,
                out_ch=w5,
                transformer_dim=t2,
                num_heads=_pick_heads(t2, preferred=[8, 6, 4, 3, 2, 1]),
                depth=d2,
                dropout=float(dropout),
            ),
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(p=float(dropout)),
            nn.Linear(w5, int(num_classes)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return self.head(x)


def build_mobilevit_classifier(
    *,
    in_channels: int,
    num_classes: int,
    image_size: int = 64,
    variant: str = "mobilevit_tiny",
    width_mult: float = 1.0,
    dropout: float = 0.1,
) -> nn.Module:
    return MobileViTClassifier(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        image_size=int(image_size),
        variant=str(variant),
        width_mult=float(width_mult),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    for v in ["mobilevit_tiny", "mobilevit_small", "mobilevit_base"]:
        m = build_mobilevit_classifier(in_channels=3, num_classes=10, image_size=64, variant=v, width_mult=0.5)
        y = m(x)
        print(v, tuple(y.shape))

