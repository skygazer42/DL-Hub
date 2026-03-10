import torch
import torch.nn.functional as F
from torch import nn

from dlhub.vision.backbones._blocks import ConvBNAct, scale_channels


class _BackboneStride4(nn.Module):
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
            ConvBNAct(c_in, stem, kernel_size=3, stride=2, act="relu"),
            ConvBNAct(stem, feat, kernel_size=3, stride=2, act="relu"),
        ]
        for _ in range(d):
            layers.append(ConvBNAct(feat, feat, kernel_size=3, stride=1, act="relu"))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RFCNDetector(nn.Module):
    """R-FCN-style detector (toy-first).

    Uses position-sensitive score maps; we approximate ROI pooling by global averaging.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        stem_channels: int = 32,
        feat_channels: int = 96,
        backbone_depth: int = 2,
        k: int = 3,
    ) -> None:
        super().__init__()
        nc = int(num_classes)
        if nc <= 0:
            raise ValueError("num_classes must be > 0")
        kk = int(k)
        if kk <= 0:
            raise ValueError("k must be > 0")

        self.backbone = _BackboneStride4(
            in_channels=int(in_channels),
            stem_channels=int(stem_channels),
            feat_channels=int(feat_channels),
            depth=int(backbone_depth),
        )
        c = int(feat_channels)
        self.ps_cls = nn.Conv2d(c, nc * kk * kk, kernel_size=1)
        self.ps_box = nn.Conv2d(c, 4 * kk * kk, kernel_size=1)
        self.k = kk
        self.num_classes = nc

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = x.to(torch.float32)
        if x.ndim != 4:
            raise ValueError(f"Expected input shape (B, C, H, W), got {tuple(x.shape)}")
        feat = self.backbone(x)
        cls_map = self.ps_cls(feat)  # (B, C*k*k, H, W)
        box_map = self.ps_box(feat)  # (B, 4*k*k, H, W)

        b = feat.shape[0]
        pooled_cls = (
            F.adaptive_avg_pool2d(cls_map, (1, 1))
            .view(b, self.num_classes, self.k, self.k)
            .mean(dim=(2, 3))
        )
        pooled_box = (
            F.adaptive_avg_pool2d(box_map, (1, 1)).view(b, 4, self.k, self.k).mean(dim=(2, 3))
        )
        return {
            "ps_cls_logits": pooled_cls,
            "ps_bbox": torch.sigmoid(pooled_box),
            "ps_cls_map": cls_map,
            "ps_box_map": box_map,
        }


_VARIANTS: dict[str, dict] = {
    "rfcn_tiny": {"stem": 24, "feat": 64, "depth": 1, "k": 3},
    "rfcn_small": {"stem": 32, "feat": 96, "depth": 2, "k": 3},
    "rfcn_base": {"stem": 48, "feat": 128, "depth": 3, "k": 3},
}


def build_rfcn_detector(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "rfcn_tiny",
    width_mult: float = 1.0,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown R-FCN variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    stem = scale_channels(int(spec["stem"]), float(width_mult), min_ch=16, divisor=8)
    feat = scale_channels(int(spec["feat"]), float(width_mult), min_ch=16, divisor=8)
    return RFCNDetector(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        stem_channels=int(stem),
        feat_channels=int(feat),
        backbone_depth=int(spec["depth"]),
        k=int(spec["k"]),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 128, 128)
    m = build_rfcn_detector(in_channels=3, num_classes=2, variant="rfcn_tiny", width_mult=0.5)
    out = m(x)
    print("rfcn_tiny", {k: tuple(v.shape) for k, v in out.items()})
    loss = sum(v.mean() for v in out.values())
    loss.backward()
    print("ok")
