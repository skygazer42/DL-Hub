
import torch
from torch import nn
import torch.nn.functional as F

from dlhub.vision.backbones._blocks import ConvBNAct, scale_channels


class _SimpleBackbone(nn.Module):
    """Backbone that outputs a stride-8 feature map for PSPNet."""

    def __init__(
        self,
        *,
        in_channels: int,
        stem_channels: int,
        feat_channels: int,
        depth: int,
    ) -> None:
        super().__init__()
        c_in = int(in_channels)
        stem = int(stem_channels)
        feat = int(feat_channels)
        d = int(depth)
        if d <= 0:
            raise ValueError("depth must be > 0")

        self.stem = nn.Sequential(
            ConvBNAct(c_in, stem, kernel_size=3, stride=2, act="relu"),  # /2
            ConvBNAct(stem, stem, kernel_size=3, stride=2, act="relu"),  # /4
            ConvBNAct(stem, feat, kernel_size=3, stride=2, act="relu"),  # /8
        )
        blocks: list[nn.Module] = []
        for _ in range(d):
            blocks.append(ConvBNAct(feat, feat, kernel_size=3, stride=1, act="relu"))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        return self.blocks(x)


class PyramidPooling(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bins: tuple[int, ...] = (1, 2, 3, 6),
    ) -> None:
        super().__init__()
        c_in = int(in_channels)
        c_out = int(out_channels)
        self.bins = tuple(int(b) for b in bins)
        if any(b <= 0 for b in self.bins):
            raise ValueError("bins must be positive")

        # Each pooled branch projects to reduced channels.
        reduced = max(8, c_out // 4)
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.AdaptiveAvgPool2d((b, b)),
                    nn.Conv2d(c_in, reduced, kernel_size=1, bias=False),
                    nn.BatchNorm2d(reduced),
                    nn.ReLU(inplace=True),
                )
                for b in self.bins
            ]
        )

        concat_ch = c_in + reduced * len(self.bins)
        self.fuse = nn.Sequential(
            nn.Conv2d(concat_ch, c_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        pooled: list[torch.Tensor] = [x]
        for br in self.branches:
            y = br(x)
            y = F.interpolate(y, size=(h, w), mode="bilinear", align_corners=False)
            pooled.append(y)
        return self.fuse(torch.cat(pooled, dim=1))


class PSPNet(nn.Module):
    """PSPNet semantic segmentation (toy-first)."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        stem_channels: int = 32,
        feat_channels: int = 128,
        depth: int = 2,
        ppm_channels: int = 128,
        bins: tuple[int, ...] = (1, 2, 3, 6),
    ) -> None:
        super().__init__()
        c_in = int(in_channels)
        nc = int(num_classes)
        if nc <= 0:
            raise ValueError("num_classes must be > 0")

        self.backbone = _SimpleBackbone(
            in_channels=c_in,
            stem_channels=int(stem_channels),
            feat_channels=int(feat_channels),
            depth=int(depth),
        )
        self.ppm = PyramidPooling(int(feat_channels), int(ppm_channels), bins=tuple(bins))
        self.classifier = nn.Sequential(
            ConvBNAct(int(ppm_channels), int(ppm_channels), kernel_size=3, stride=1, act="relu"),
            nn.Conv2d(int(ppm_channels), nc, kernel_size=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        if x.ndim != 4:
            raise ValueError(f"Expected input shape (B, C, H, W), got {tuple(x.shape)}")

        inp_h, inp_w = x.shape[-2:]
        feats = self.backbone(x)
        feats = self.ppm(feats)
        logits = self.classifier(feats)
        return F.interpolate(logits, size=(inp_h, inp_w), mode="bilinear", align_corners=False)


_VARIANTS: dict[str, dict] = {
    "pspnet_tiny": {"stem": 24, "feat": 96, "depth": 1, "ppm": 96},
    "pspnet_small": {"stem": 32, "feat": 128, "depth": 2, "ppm": 128},
    "pspnet_base": {"stem": 48, "feat": 192, "depth": 3, "ppm": 192},
}


def build_pspnet_segmenter(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "pspnet_small",
    width_mult: float = 1.0,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown PSPNet variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]

    stem = scale_channels(int(spec["stem"]), float(width_mult), min_ch=16, divisor=8)
    feat = scale_channels(int(spec["feat"]), float(width_mult), min_ch=16, divisor=8)
    ppm = scale_channels(int(spec["ppm"]), float(width_mult), min_ch=16, divisor=8)

    return PSPNet(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        stem_channels=int(stem),
        feat_channels=int(feat),
        depth=int(spec["depth"]),
        ppm_channels=int(ppm),
        bins=(1, 2, 3, 6),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_pspnet_segmenter(in_channels=3, num_classes=2, variant="pspnet_tiny")
    y = m(x)
    print("pspnet_tiny", tuple(y.shape))
    loss = y.mean()
    loss.backward()
    print("ok")

