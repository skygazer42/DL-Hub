import torch
from torch import nn

from dlhub.vision.backbones._blocks import DropPath, GlobalAvgPoolHead, scale_channels
from dlhub.vision.backbones._transformer import TransformerEncoderBlock


class ConvFFN(nn.Module):
    def __init__(self, dim: int, *, mlp_ratio: float = 4.0, drop_path: float = 0.0) -> None:
        super().__init__()
        d = int(dim)
        hidden = int(round(d * float(mlp_ratio)))
        self.dw = nn.Conv2d(d, d, kernel_size=3, padding=1, groups=d, bias=False)
        self.pw1 = nn.Conv2d(d, hidden, kernel_size=1, bias=True)
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(hidden, d, kernel_size=1, bias=True)
        self.drop_path = DropPath(float(drop_path))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.dw(x)
        y = self.pw1(y)
        y = self.act(y)
        y = self.pw2(y)
        return x + self.drop_path(y)


class UniFormerV2Stage(nn.Module):
    def __init__(
        self, *, dim: int, depth: int, num_heads: int, mode: str, drop_path: float
    ) -> None:
        super().__init__()
        d = int(dim)
        depth = int(depth)
        mode = str(mode).lower().strip()
        blocks: list[nn.Module] = []
        if mode == "conv":
            for i in range(depth):
                blocks.append(
                    ConvFFN(d, mlp_ratio=4.0, drop_path=float(drop_path) * (i / max(1, depth - 1)))
                )
        elif mode == "attn":
            for i in range(depth):
                blocks.append(
                    TransformerEncoderBlock(
                        d,
                        int(num_heads),
                        mlp_ratio=4.0,
                        dropout=0.0,
                        drop_path=float(drop_path) * (i / max(1, depth - 1)),
                    )
                )
        else:
            raise ValueError("mode must be 'conv' or 'attn'")
        self.blocks = nn.Sequential(*blocks)
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "conv":
            return self.blocks(x)
        b, c, h, w = x.shape
        t = x.flatten(2).transpose(1, 2)
        t = self.blocks(t)
        return t.transpose(1, 2).contiguous().view(b, c, h, w)


class UniFormerV2Classifier(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        dims: tuple[int, int, int, int] = (64, 128, 320, 512),
        depths: tuple[int, int, int, int] = (2, 2, 6, 2),
        heads: tuple[int, int, int, int] = (1, 2, 5, 8),
        width_mult: float = 1.0,
        drop_path: float = 0.1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        dims = tuple(scale_channels(int(d), float(width_mult), min_ch=16, divisor=8) for d in dims)
        depths = tuple(int(x) for x in depths)
        heads = tuple(int(h) for h in heads)

        self.stem = nn.Sequential(
            nn.Conv2d(int(in_channels), dims[0], kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(dims[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(dims[0], dims[1], kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(dims[1]),
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(dims[1], dims[2], kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(dims[2]),
        )
        self.down4 = nn.Sequential(
            nn.Conv2d(dims[2], dims[3], kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(dims[3]),
        )

        self.stage1 = UniFormerV2Stage(
            dim=dims[0],
            depth=depths[0],
            num_heads=heads[0],
            mode="conv",
            drop_path=float(drop_path),
        )
        self.stage2 = UniFormerV2Stage(
            dim=dims[1],
            depth=depths[1],
            num_heads=heads[1],
            mode="conv",
            drop_path=float(drop_path),
        )
        self.stage3 = UniFormerV2Stage(
            dim=dims[2],
            depth=depths[2],
            num_heads=heads[2],
            mode="attn",
            drop_path=float(drop_path),
        )
        self.stage4 = UniFormerV2Stage(
            dim=dims[3],
            depth=depths[3],
            num_heads=heads[3],
            mode="attn",
            drop_path=float(drop_path),
        )

        self.head = GlobalAvgPoolHead(dims[-1], int(num_classes), dropout=float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        x = self.stem(x)
        x = self.stage1(x)
        x = self.down2(x)
        x = self.stage2(x)
        x = self.down3(x)
        x = self.stage3(x)
        x = self.down4(x)
        x = self.stage4(x)
        return self.head(x)


_VARIANTS: dict[str, dict] = {
    "uniformer_v2_s": {"dims": (64, 128, 320, 512), "depths": (2, 2, 6, 2), "heads": (1, 2, 5, 8)},
    "uniformer_v2_b": {"dims": (64, 128, 320, 512), "depths": (3, 4, 8, 3), "heads": (1, 2, 5, 8)},
}


def build_uniformer_v2_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "uniformer_v2_s",
    width_mult: float = 1.0,
    drop_path: float = 0.1,
    dropout: float = 0.1,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(
            f"Unknown UniFormerV2 variant: {variant!r}. Supported: {sorted(_VARIANTS)}"
        )
    spec = _VARIANTS[name]
    return UniFormerV2Classifier(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        dims=tuple(map(int, spec["dims"])),
        depths=tuple(map(int, spec["depths"])),
        heads=tuple(map(int, spec["heads"])),
        width_mult=float(width_mult),
        drop_path=float(drop_path),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_uniformer_v2_classifier(
        in_channels=3, num_classes=10, variant="uniformer_v2_s", width_mult=0.5
    )
    y = m(x)
    print("uniformer_v2_s", tuple(y.shape))
