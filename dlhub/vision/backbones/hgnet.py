import torch
from torch import nn

from dlhub.vision.backbones._blocks import (
    ChannelShuffle,
    ConvBNAct,
    GlobalAvgPoolHead,
    SqueezeExcite,
    scale_channels,
)


class HGUnit(nn.Module):
    """HGNet-style lightweight unit (simplified): grouped conv + channel shuffle + SE."""

    def __init__(
        self, in_ch: int, out_ch: int, *, stride: int, groups: int, se: bool = True
    ) -> None:
        super().__init__()
        c_in = int(in_ch)
        c_out = int(out_ch)
        s = int(stride)
        g = int(groups)
        if c_out % g != 0:
            # fall back to a safe divisor
            g = 1
        self.proj = ConvBNAct(c_in, c_out, kernel_size=1, stride=1, padding=0, act="relu")
        self.gconv = ConvBNAct(c_out, c_out, kernel_size=3, stride=s, groups=g, act="relu")
        self.shuffle = ChannelShuffle(g) if g > 1 else nn.Identity()
        self.pw = ConvBNAct(c_out, c_out, kernel_size=1, stride=1, padding=0, act="relu")
        self.se = SqueezeExcite(c_out, se_ratio=0.25) if bool(se) else nn.Identity()
        self.use_res = s == 1 and c_in == c_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.proj(x)
        x = self.gconv(x)
        x = self.shuffle(x)
        x = self.pw(x)
        x = self.se(x)
        if self.use_res:
            x = x + identity
        return x


class HGNetClassifier(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        dims: tuple[int, int, int, int],
        depths: tuple[int, int, int, int],
        groups: int = 4,
        width_mult: float = 1.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        dims = tuple(scale_channels(int(d), float(width_mult), min_ch=16, divisor=8) for d in dims)
        depths = tuple(int(d) for d in depths)

        self.stem = nn.Sequential(
            ConvBNAct(int(in_channels), dims[0], kernel_size=3, stride=2, act="relu"),
            ConvBNAct(dims[0], dims[0], kernel_size=3, stride=1, act="relu"),
        )

        def make_stage(in_ch: int, out_ch: int, depth: int, *, stride: int) -> nn.Sequential:
            blocks: list[nn.Module] = [
                HGUnit(in_ch, out_ch, stride=int(stride), groups=int(groups), se=True)
            ]
            for _ in range(int(depth) - 1):
                blocks.append(HGUnit(out_ch, out_ch, stride=1, groups=int(groups), se=True))
            return nn.Sequential(*blocks)

        self.stage1 = make_stage(dims[0], dims[0], depths[0], stride=1)
        self.stage2 = make_stage(dims[0], dims[1], depths[1], stride=2)
        self.stage3 = make_stage(dims[1], dims[2], depths[2], stride=2)
        self.stage4 = make_stage(dims[2], dims[3], depths[3], stride=2)

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
    "hgnet_tiny": {"dims": (32, 64, 128, 256), "depths": (1, 2, 4, 2), "groups": 4},
    "hgnet_small": {"dims": (48, 96, 192, 384), "depths": (2, 2, 6, 2), "groups": 8},
    "hgnet_base": {"dims": (64, 128, 256, 512), "depths": (2, 3, 8, 3), "groups": 8},
    # "PP-HGNet" naming is common in some repos; we treat it as a variant alias here.
    "pphgnet_tiny": {"dims": (32, 64, 128, 256), "depths": (2, 2, 6, 2), "groups": 8},
    "pphgnet_base": {"dims": (64, 128, 256, 512), "depths": (3, 4, 10, 3), "groups": 8},
}


def build_hgnet_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "hgnet_tiny",
    width_mult: float = 1.0,
    dropout: float = 0.1,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown HGNet variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    return HGNetClassifier(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        dims=tuple(map(int, spec["dims"])),
        depths=tuple(map(int, spec["depths"])),
        groups=int(spec["groups"]),
        width_mult=float(width_mult),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_hgnet_classifier(in_channels=3, num_classes=10, variant="hgnet_tiny", width_mult=0.5)
    y = m(x)
    print("hgnet_tiny", tuple(y.shape))
