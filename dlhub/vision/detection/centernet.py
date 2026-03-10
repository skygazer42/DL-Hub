import torch
from torch import nn

from dlhub.vision.backbones._blocks import ConvBNAct, scale_channels


class _SimpleEncoder(nn.Module):
    """A tiny encoder that produces a stride-4 feature map."""

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

        layers: list[nn.Module] = [
            ConvBNAct(c_in, stem, kernel_size=3, stride=2, act="relu"),
            ConvBNAct(stem, feat, kernel_size=3, stride=2, act="relu"),
        ]
        for _ in range(d):
            layers.append(ConvBNAct(feat, feat, kernel_size=3, stride=1, act="relu"))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CenterNetDetector(nn.Module):
    """CenterNet-style keypoint detector (toy-first).

    Outputs (raw, stride=4):
    - heatmap: (B, num_classes, H/4, W/4) logits (use sigmoid in loss/inference)
    - wh: (B, 2, H/4, W/4) predicted width/height (non-negative)
    - offset: (B, 2, H/4, W/4) sub-pixel offsets in [~ -0.5, 0.5]
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
        c_in = int(in_channels)
        nc = int(num_classes)
        if nc <= 0:
            raise ValueError("num_classes must be > 0")

        self.stride = 4
        self.encoder = _SimpleEncoder(
            in_channels=c_in,
            stem_channels=int(stem_channels),
            feat_channels=int(feat_channels),
            depth=int(depth),
        )

        hc = int(head_channels)
        self.hm_head = nn.Sequential(
            ConvBNAct(int(feat_channels), hc, kernel_size=3, stride=1, act="relu"),
            nn.Conv2d(hc, nc, kernel_size=1, bias=True),
        )
        self.wh_head = nn.Sequential(
            ConvBNAct(int(feat_channels), hc, kernel_size=3, stride=1, act="relu"),
            nn.Conv2d(hc, 2, kernel_size=1, bias=True),
        )
        self.off_head = nn.Sequential(
            ConvBNAct(int(feat_channels), hc, kernel_size=3, stride=1, act="relu"),
            nn.Conv2d(hc, 2, kernel_size=1, bias=True),
        )

        # Initialize heatmap bias to produce low confidence initially.
        last = self.hm_head[-1]
        if isinstance(last, nn.Conv2d):
            nn.init.constant_(last.bias, -2.19)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = x.to(torch.float32)
        if x.ndim != 4:
            raise ValueError(f"Expected input shape (B, C, H, W), got {tuple(x.shape)}")

        feats = self.encoder(x)
        heatmap = self.hm_head(feats)
        wh = torch.relu(self.wh_head(feats))
        offset = self.off_head(feats)
        return {"heatmap": heatmap, "wh": wh, "offset": offset}


_VARIANTS: dict[str, dict] = {
    "centernet_tiny": {"stem": 24, "feat": 48, "depth": 1, "head": 48},
    "centernet_small": {"stem": 32, "feat": 64, "depth": 2, "head": 64},
    "centernet_base": {"stem": 48, "feat": 96, "depth": 3, "head": 96},
}


def build_centernet_detector(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "centernet_tiny",
    width_mult: float = 1.0,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown CenterNet variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    stem = scale_channels(int(spec["stem"]), float(width_mult), min_ch=16, divisor=8)
    feat = scale_channels(int(spec["feat"]), float(width_mult), min_ch=16, divisor=8)
    head = scale_channels(int(spec["head"]), float(width_mult), min_ch=16, divisor=8)
    return CenterNetDetector(
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
    m = build_centernet_detector(in_channels=3, num_classes=2, variant="centernet_tiny")
    out = m(x)
    print("centernet_tiny", {k: tuple(v.shape) for k, v in out.items()})
    loss = sum(v.mean() for v in out.values())
    loss.backward()
    print("ok")
