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
    """MBConv (EfficientNet-style), simplified."""

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
        layers.append(SqueezeExcite(hidden, se_ratio=float(se_ratio)))
        layers.append(nn.Conv2d(hidden, c_out, kernel_size=1, bias=False))
        layers.append(nn.BatchNorm2d(c_out))
        self.block = nn.Sequential(*layers)
        self.drop = nn.Dropout2d(p=float(dropout)) if float(dropout) > 0 else nn.Identity()
        self.drop_path = DropPath(float(drop_path))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.block(x)
        y = self.drop(y)
        if self.use_res:
            y = self.drop_path(y)
            return x + y
        return y


class Attention2DBlock(nn.Module):
    """Transformer encoder applied to flattened spatial tokens."""

    def __init__(self, dim: int, *, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.dim = int(dim)
        self.block = TransformerEncoderBlock(
            dim=int(dim), num_heads=int(num_heads), dropout=float(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        b, c, h, w = x.shape
        if c != self.dim:
            raise ValueError(f"Expected channels={self.dim}, got {c}")
        t = x.flatten(2).transpose(1, 2)  # (B, T, C)
        t = self.block(t)
        return t.transpose(1, 2).contiguous().view(b, c, h, w)


class CoAtNetClassifier(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        image_size: int = 64,
        variant: str = "coatnet_tiny",
        width_mult: float = 1.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if int(image_size) % 16 != 0:
            raise ValueError("CoAtNet expects image_size divisible by 16")

        name = str(variant).lower().strip()
        w = float(width_mult)
        if name in {"coatnet_tiny", "tiny", "coatnet"}:
            widths = (32, 64, 128, 256)
            depths = (2, 2, 2, 2)
        elif name in {"coatnet_small", "small"}:
            widths = (32, 96, 192, 384)
            depths = (2, 3, 3, 3)
        elif name in {"coatnet_base", "base"}:
            widths = (48, 128, 256, 512)
            depths = (3, 4, 4, 4)
        else:
            raise ValueError(
                "Unknown CoAtNet variant. Supported: coatnet_tiny|coatnet_small|coatnet_base"
            )

        w1, w2, w3, w4 = (scale_channels(int(c), w, min_ch=32, divisor=8) for c in widths)
        d1, d2, d3, d4 = (int(d) for d in depths)

        self.stem = ConvBNAct(int(in_channels), w1, kernel_size=3, stride=2, act="silu")

        self.stage1 = nn.Sequential(
            *[MBConv(w1, w1, stride=1, expand_ratio=2, dropout=float(dropout)) for _ in range(d1)]
        )

        s2: list[nn.Module] = [MBConv(w1, w2, stride=2, expand_ratio=2, dropout=float(dropout))]
        s2.extend(
            [
                MBConv(w2, w2, stride=1, expand_ratio=2, dropout=float(dropout))
                for _ in range(d2 - 1)
            ]
        )
        self.stage2 = nn.Sequential(*s2)

        heads3 = _pick_heads(w3, preferred=[12, 8, 6, 4, 3, 2, 1])
        heads4 = _pick_heads(w4, preferred=[12, 8, 6, 4, 3, 2, 1])

        s3: list[nn.Module] = [ConvBNAct(w2, w3, kernel_size=3, stride=2, act="silu")]
        s3.extend(
            [Attention2DBlock(w3, num_heads=heads3, dropout=float(dropout)) for _ in range(d3)]
        )
        self.stage3 = nn.Sequential(*s3)

        s4: list[nn.Module] = [ConvBNAct(w3, w4, kernel_size=3, stride=2, act="silu")]
        s4.extend(
            [Attention2DBlock(w4, num_heads=heads4, dropout=float(dropout)) for _ in range(d4)]
        )
        self.stage4 = nn.Sequential(*s4)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(p=float(dropout)),
            nn.Linear(w4, int(num_classes)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return self.head(x)


def build_coatnet_classifier(
    *,
    in_channels: int,
    num_classes: int,
    image_size: int = 64,
    variant: str = "coatnet_tiny",
    width_mult: float = 1.0,
    dropout: float = 0.1,
) -> nn.Module:
    return CoAtNetClassifier(
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
    for v in ["coatnet_tiny", "coatnet_small", "coatnet_base"]:
        m = build_coatnet_classifier(
            in_channels=3, num_classes=10, image_size=64, variant=v, width_mult=0.5
        )
        y = m(x)
        print(v, tuple(y.shape))
