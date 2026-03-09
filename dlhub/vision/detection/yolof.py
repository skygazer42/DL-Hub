
import torch
from torch import nn
import torch.nn.functional as F

from dlhub.vision.backbones._blocks import ConvBNAct, scale_channels
from dlhub.vision.detection._common import BackboneC3C5, check_nchw


class _DilatedEncoder(nn.Module):
    def __init__(self, channels: int, *, num_layers: int = 4) -> None:
        super().__init__()
        c = int(channels)
        n = int(num_layers)
        if n <= 0:
            raise ValueError("num_layers must be > 0")
        layers: list[nn.Module] = []
        for i in range(n):
            d = 1 if i == 0 else 2**i
            layers.append(
                nn.Sequential(
                    nn.Conv2d(c, c, kernel_size=3, padding=d, dilation=d, bias=False),
                    nn.BatchNorm2d(c),
                    nn.ReLU(inplace=True),
                )
            )
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class YOLOFDetector(nn.Module):
    """YOLOF-style one-level detector (toy-first).

    Uses only the C5 feature with a dilated encoder.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        stem_channels: int = 32,
        backbone_channels: tuple[int, int, int] = (64, 96, 128),
        backbone_depth: int = 2,
        feat_channels: int = 128,
        encoder_layers: int = 4,
        num_anchors: int = 9,
        head_convs: int = 2,
    ) -> None:
        super().__init__()
        nc = int(num_classes)
        if nc <= 0:
            raise ValueError("num_classes must be > 0")
        c3, c4, c5 = (int(x) for x in backbone_channels)
        self.backbone = BackboneC3C5(
            in_channels=int(in_channels),
            stem_channels=int(stem_channels),
            c3_channels=c3,
            c4_channels=c4,
            c5_channels=c5,
            depth=int(backbone_depth),
            act="relu",
        )
        feat = int(feat_channels)
        self.proj = nn.Conv2d(c5, feat, kernel_size=1)
        self.encoder = _DilatedEncoder(feat, num_layers=int(encoder_layers))

        tower = nn.Sequential(*[ConvBNAct(feat, feat, kernel_size=3, stride=1, act="relu") for _ in range(int(head_convs))])
        self.cls_tower = tower
        self.box_tower = tower
        na = int(num_anchors)
        self.cls = nn.Conv2d(feat, na * nc, kernel_size=1)
        self.box = nn.Conv2d(feat, na * 4, kernel_size=1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = check_nchw(x)
        _, _, c5 = self.backbone(x)
        f = self.encoder(F.relu(self.proj(c5)))
        return {"cls_logits": self.cls(self.cls_tower(f)), "bbox_deltas": self.box(self.box_tower(f))}


_VARIANTS: dict[str, dict] = {
    "yolof_tiny": {"stem": 24, "c3": 48, "c4": 64, "c5": 80, "depth": 1, "feat": 96, "enc": 3, "head": 1},
    "yolof_small": {"stem": 32, "c3": 64, "c4": 96, "c5": 128, "depth": 2, "feat": 128, "enc": 4, "head": 2},
    "yolof_base": {"stem": 48, "c3": 96, "c4": 144, "c5": 192, "depth": 3, "feat": 192, "enc": 4, "head": 2},
}


def build_yolof_detector(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "yolof_tiny",
    width_mult: float = 1.0,
    num_anchors: int = 9,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown YOLOF variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]

    stem = scale_channels(int(spec["stem"]), float(width_mult), min_ch=16, divisor=8)
    c3 = scale_channels(int(spec["c3"]), float(width_mult), min_ch=16, divisor=8)
    c4 = scale_channels(int(spec["c4"]), float(width_mult), min_ch=16, divisor=8)
    c5 = scale_channels(int(spec["c5"]), float(width_mult), min_ch=16, divisor=8)
    feat = scale_channels(int(spec["feat"]), float(width_mult), min_ch=16, divisor=8)

    return YOLOFDetector(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        stem_channels=int(stem),
        backbone_channels=(int(c3), int(c4), int(c5)),
        backbone_depth=int(spec["depth"]),
        feat_channels=int(feat),
        encoder_layers=int(spec["enc"]),
        num_anchors=int(num_anchors),
        head_convs=int(spec["head"]),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 128, 128)
    m = build_yolof_detector(in_channels=3, num_classes=3, variant="yolof_tiny", width_mult=0.5)
    out = m(x)
    print("yolof_tiny", {k: tuple(v.shape) for k, v in out.items()})
    loss = sum(v.mean() for v in out.values())
    loss.backward()
    print("ok")

