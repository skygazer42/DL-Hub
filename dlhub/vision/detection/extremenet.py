import torch
from torch import nn

from dlhub.vision.backbones._blocks import ConvBNAct, scale_channels


class _EncoderStride4(nn.Module):
    def __init__(
        self, *, in_channels: int, stem_channels: int, feat_channels: int, depth: int
    ) -> None:
        super().__init__()
        c_in = int(in_channels)
        stem = int(stem_channels)
        feat = int(feat_channels)
        d = int(depth)
        if d <= 0:
            raise ValueError("depth must be > 0")
        layers: list[nn.Module] = [
            ConvBNAct(c_in, stem, kernel_size=3, stride=2, act="relu"),  # /2
            ConvBNAct(stem, feat, kernel_size=3, stride=2, act="relu"),  # /4
        ]
        for _ in range(d):
            layers.append(ConvBNAct(feat, feat, kernel_size=3, stride=1, act="relu"))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ExtremeNetDetector(nn.Module):
    """ExtremeNet-style keypoint detector (compact-first).

    Predicts heatmaps for extreme points (left/right/top/bottom) and center.
    Output (raw, stride=4):
    - left/right/top/bottom/center: (B, C, H/4, W/4)
    - offset: (B, 2, H/4, W/4)
    """

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        stem_channels: int = 32,
        feat_channels: int = 64,
        depth: int = 2,
        head_channels: int = 64,
    ) -> None:
        super().__init__()
        nc = int(num_classes)
        if nc <= 0:
            raise ValueError("num_classes must be > 0")
        self.stride = 4
        self.encoder = _EncoderStride4(
            in_channels=int(in_channels),
            stem_channels=int(stem_channels),
            feat_channels=int(feat_channels),
            depth=int(depth),
        )
        hc = int(head_channels)

        def hm_head() -> nn.Sequential:
            return nn.Sequential(
                ConvBNAct(int(feat_channels), hc, kernel_size=3, stride=1, act="relu"),
                nn.Conv2d(hc, nc, kernel_size=1, bias=True),
            )

        self.left = hm_head()
        self.right = hm_head()
        self.top = hm_head()
        self.bottom = hm_head()
        self.center = hm_head()
        self.offset = nn.Sequential(
            ConvBNAct(int(feat_channels), hc, kernel_size=3, stride=1, act="relu"),
            nn.Conv2d(hc, 2, kernel_size=1, bias=True),
        )

        for m in [self.left[-1], self.right[-1], self.top[-1], self.bottom[-1], self.center[-1]]:
            if isinstance(m, nn.Conv2d):
                nn.init.constant_(m.bias, -2.19)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = x.to(torch.float32)
        if x.ndim != 4:
            raise ValueError(f"Expected input shape (B, C, H, W), got {tuple(x.shape)}")
        f = self.encoder(x)
        return {
            "left": self.left(f),
            "right": self.right(f),
            "top": self.top(f),
            "bottom": self.bottom(f),
            "center": self.center(f),
            "offset": self.offset(f),
        }


_VARIANTS: dict[str, dict] = {
    "extremenet_tiny": {"stem": 24, "feat": 48, "depth": 1, "head": 48},
    "extremenet_small": {"stem": 32, "feat": 64, "depth": 2, "head": 64},
    "extremenet_base": {"stem": 48, "feat": 96, "depth": 3, "head": 96},
}


def build_extremenet_detector(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "extremenet_tiny",
    width_mult: float = 1.0,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown ExtremeNet variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    stem = scale_channels(int(spec["stem"]), float(width_mult), min_ch=16, divisor=8)
    feat = scale_channels(int(spec["feat"]), float(width_mult), min_ch=16, divisor=8)
    head = scale_channels(int(spec["head"]), float(width_mult), min_ch=16, divisor=8)
    return ExtremeNetDetector(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        stem_channels=int(stem),
        feat_channels=int(feat),
        depth=int(spec["depth"]),
        head_channels=int(head),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_extremenet_detector(
        in_channels=3, num_classes=2, variant="extremenet_tiny", width_mult=0.5
    )
    out = m(x)
    print("extremenet_tiny", {k: tuple(v.shape) for k, v in out.items()})
    loss = sum(v.mean() for v in out.values())
    loss.backward()
    print("ok")
