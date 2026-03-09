
import torch
from torch import nn

from dlhub.vision.backbones._blocks import ConvBNAct, GlobalAvgPoolHead, SqueezeExcite, scale_channels


class OSABlock(nn.Module):
    """VoVNet OSA (One-Shot Aggregation) block (simplified)."""

    def __init__(
        self,
        in_ch: int,
        mid_ch: int,
        out_ch: int,
        *,
        num_layers: int = 3,
        use_se: bool = True,
        residual: bool = False,
    ) -> None:
        super().__init__()
        c_in = int(in_ch)
        c_mid = int(mid_ch)
        c_out = int(out_ch)
        n = int(num_layers)
        if n <= 0:
            raise ValueError("num_layers must be > 0")

        convs: list[nn.Module] = []
        cur = c_in
        for _ in range(n):
            convs.append(ConvBNAct(cur, c_mid, kernel_size=3, stride=1, act="relu"))
            cur = c_mid
        self.convs = nn.ModuleList(convs)
        self.concat = ConvBNAct(c_in + n * c_mid, c_out, kernel_size=1, stride=1, padding=0, act="relu")
        self.se = SqueezeExcite(c_out, se_ratio=0.25) if bool(use_se) else nn.Identity()
        self.residual = bool(residual)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [x]
        y = x
        for conv in self.convs:
            y = conv(y)
            feats.append(y)
        y = torch.cat(feats, dim=1)
        y = self.concat(y)
        y = self.se(y)
        if self.residual and y.shape == x.shape:
            y = y + x
        return y


class VoVNetClassifier(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        dims: tuple[int, int, int, int],
        depths: tuple[int, int, int, int],
        mid_ratio: float = 0.5,
        width_mult: float = 1.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        dims = tuple(scale_channels(int(d), float(width_mult), min_ch=16, divisor=8) for d in dims)
        depths = tuple(int(d) for d in depths)

        stem = [
            ConvBNAct(int(in_channels), dims[0] // 2, kernel_size=3, stride=2, act="relu"),
            ConvBNAct(dims[0] // 2, dims[0] // 2, kernel_size=3, stride=1, act="relu"),
            ConvBNAct(dims[0] // 2, dims[0], kernel_size=3, stride=2, act="relu"),
        ]
        self.stem = nn.Sequential(*stem)

        def make_stage(in_ch: int, out_ch: int, depth: int) -> nn.Sequential:
            blocks: list[nn.Module] = []
            mid = max(16, int(round(float(out_ch) * float(mid_ratio))))
            blocks.append(OSABlock(in_ch, mid, out_ch, num_layers=3, use_se=True, residual=False))
            for _ in range(int(depth) - 1):
                blocks.append(OSABlock(out_ch, mid, out_ch, num_layers=3, use_se=True, residual=True))
            return nn.Sequential(*blocks)

        self.down2 = ConvBNAct(dims[0], dims[1], kernel_size=3, stride=2, act="relu")
        self.stage2 = make_stage(dims[1], dims[1], depths[1])

        self.down3 = ConvBNAct(dims[1], dims[2], kernel_size=3, stride=2, act="relu")
        self.stage3 = make_stage(dims[2], dims[2], depths[2])

        self.down4 = ConvBNAct(dims[2], dims[3], kernel_size=3, stride=2, act="relu")
        self.stage4 = make_stage(dims[3], dims[3], depths[3])

        self.head = GlobalAvgPoolHead(dims[3], int(num_classes), dropout=float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        x = self.stem(x)
        x = self.down2(x)
        x = self.stage2(x)
        x = self.down3(x)
        x = self.stage3(x)
        x = self.down4(x)
        x = self.stage4(x)
        return self.head(x)


_VARIANTS: dict[str, dict] = {
    "vovnet_tiny": {"dims": (64, 128, 256, 384), "depths": (1, 1, 2, 1), "mid_ratio": 0.5},
    "vovnet_small": {"dims": (64, 128, 256, 512), "depths": (1, 2, 4, 2), "mid_ratio": 0.5},
    "vovnet_base": {"dims": (64, 160, 320, 640), "depths": (2, 2, 6, 2), "mid_ratio": 0.5},
}


def build_vovnet_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "vovnet_tiny",
    width_mult: float = 1.0,
    dropout: float = 0.1,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown VoVNet variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    return VoVNetClassifier(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        dims=tuple(map(int, spec["dims"])),
        depths=tuple(map(int, spec["depths"])),
        mid_ratio=float(spec["mid_ratio"]),
        width_mult=float(width_mult),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_vovnet_classifier(in_channels=3, num_classes=10, variant="vovnet_tiny", width_mult=0.5)
    y = m(x)
    print("vovnet_tiny", tuple(y.shape))
