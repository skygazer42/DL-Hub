import torch
from torch import nn

from dlhub.vision.backbones._blocks import ConvBNAct, GlobalAvgPoolHead, scale_channels


class ResidualUnit(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, *, stride: int) -> None:
        super().__init__()
        self.conv1 = ConvBNAct(
            int(in_ch), int(out_ch), kernel_size=3, stride=int(stride), act="relu"
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(int(out_ch), int(out_ch), kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(int(out_ch)),
        )
        self.act = nn.ReLU(inplace=True)
        self.down: nn.Module | None = None
        if int(stride) != 1 or int(in_ch) != int(out_ch):
            self.down = nn.Sequential(
                nn.Conv2d(int(in_ch), int(out_ch), kernel_size=1, stride=int(stride), bias=False),
                nn.BatchNorm2d(int(out_ch)),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        if self.down is not None:
            identity = self.down(identity)
        return self.act(x + identity)


class AttentionBlock(nn.Module):
    """Residual Attention block (simplified).

    trunk: residual units
    mask: down->res->up->sigmoid
    output: (1 + mask) * trunk
    """

    def __init__(self, channels: int, *, depth: int = 1) -> None:
        super().__init__()
        c = int(channels)
        d = int(depth)
        self.trunk = nn.Sequential(*[ResidualUnit(c, c, stride=1) for _ in range(d)])
        self.mask_down = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2), ResidualUnit(c, c, stride=1)
        )
        self.mask_mid = nn.Sequential(*[ResidualUnit(c, c, stride=1) for _ in range(d)])
        self.mask_up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.mask_out = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        trunk = self.trunk(x)
        mask = self.mask_down(x)
        mask = self.mask_mid(mask)
        mask = self.mask_up(mask)
        if mask.shape[-2:] != trunk.shape[-2:]:
            mask = nn.functional.interpolate(
                mask, size=trunk.shape[-2:], mode="bilinear", align_corners=False
            )
        mask = self.mask_out(mask)
        return trunk * (1.0 + mask)


class ResAttNetClassifier(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        channels: tuple[int, int, int, int] = (64, 128, 256, 512),
        depths: tuple[int, int, int, int] = (1, 1, 2, 1),
        attn_depth: int = 1,
        width_mult: float = 1.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        chs = tuple(scale_channels(int(c), float(width_mult)) for c in channels)
        self.stem = nn.Sequential(
            ConvBNAct(int(in_channels), chs[0], kernel_size=3, stride=2, act="relu"),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        def make_stage(in_ch: int, out_ch: int, depth: int, *, stride: int) -> nn.Sequential:
            blocks: list[nn.Module] = []
            blocks.append(ResidualUnit(in_ch, out_ch, stride=int(stride)))
            blocks.append(AttentionBlock(out_ch, depth=int(attn_depth)))
            for _ in range(int(depth) - 1):
                blocks.append(ResidualUnit(out_ch, out_ch, stride=1))
            return nn.Sequential(*blocks)

        self.stage1 = make_stage(chs[0], chs[0], depths[0], stride=1)
        self.stage2 = make_stage(chs[0], chs[1], depths[1], stride=2)
        self.stage3 = make_stage(chs[1], chs[2], depths[2], stride=2)
        self.stage4 = make_stage(chs[2], chs[3], depths[3], stride=2)
        self.head = GlobalAvgPoolHead(chs[3], int(num_classes), dropout=float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return self.head(x)


_VARIANTS: dict[str, dict] = {
    "resattnet56": {"channels": (64, 128, 256, 512), "depths": (1, 1, 2, 1), "attn_depth": 1},
    "resattnet92": {"channels": (64, 128, 256, 512), "depths": (2, 2, 3, 2), "attn_depth": 1},
}


def build_resattnet_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "resattnet56",
    width_mult: float = 1.0,
    dropout: float = 0.1,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown ResAttNet variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    return ResAttNetClassifier(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        channels=tuple(map(int, spec["channels"])),
        depths=tuple(map(int, spec["depths"])),
        attn_depth=int(spec["attn_depth"]),
        width_mult=float(width_mult),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_resattnet_classifier(
        in_channels=3, num_classes=10, variant="resattnet56", width_mult=0.5
    )
    y = m(x)
    print("resattnet56", tuple(y.shape))
