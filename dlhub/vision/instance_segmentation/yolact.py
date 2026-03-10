import torch
from torch import nn

from dlhub.vision.backbones._blocks import ConvBNAct, scale_channels


class _BackboneLowDet(nn.Module):
    """Backbone that returns (low_stride4, det_stride8) feature maps."""

    def __init__(
        self,
        *,
        in_channels: int,
        stem_channels: int,
        low_channels: int,
        det_channels: int,
        depth: int,
    ) -> None:
        super().__init__()
        c_in = int(in_channels)
        stem = int(stem_channels)
        low = int(low_channels)
        det = int(det_channels)
        d = int(depth)
        if d <= 0:
            raise ValueError("depth must be > 0")

        self.stem = ConvBNAct(c_in, stem, kernel_size=3, stride=2, act="relu")  # /2
        self.to_low = ConvBNAct(stem, low, kernel_size=3, stride=2, act="relu")  # /4

        layers: list[nn.Module] = [ConvBNAct(low, det, kernel_size=3, stride=2, act="relu")]  # /8
        for _ in range(d - 1):
            layers.append(ConvBNAct(det, det, kernel_size=3, stride=1, act="relu"))
        self.det = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        low = self.to_low(x)
        det = self.det(low)
        return low, det


class _ProtoNet(nn.Module):
    def __init__(self, in_ch: int, proto_ch: int, *, depth: int = 3) -> None:
        super().__init__()
        c_in = int(in_ch)
        p = int(proto_ch)
        d = int(depth)
        if d <= 0:
            raise ValueError("depth must be > 0")

        layers: list[nn.Module] = []
        ch = c_in
        for _ in range(d - 1):
            layers.append(ConvBNAct(ch, ch, kernel_size=3, stride=1, act="relu"))
        layers.append(nn.Conv2d(ch, p, kernel_size=1, bias=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # raw prototypes; downstream can apply sigmoid
        return self.net(x)


class _PredictionHead(nn.Module):
    def __init__(
        self,
        *,
        in_ch: int,
        num_classes: int,
        num_anchors: int,
        num_protos: int,
        head_channels: int,
        num_convs: int,
    ) -> None:
        super().__init__()
        c_in = int(in_ch)
        nc = int(num_classes)
        na = int(num_anchors)
        np = int(num_protos)
        hc = int(head_channels)
        n = int(num_convs)
        if nc <= 0:
            raise ValueError("num_classes must be > 0")
        if na <= 0:
            raise ValueError("num_anchors must be > 0")
        if np <= 0:
            raise ValueError("num_protos must be > 0")
        if n <= 0:
            raise ValueError("num_convs must be > 0")

        layers: list[nn.Module] = [ConvBNAct(c_in, hc, kernel_size=3, stride=1, act="relu")]
        for _ in range(n - 1):
            layers.append(ConvBNAct(hc, hc, kernel_size=3, stride=1, act="relu"))
        self.tower = nn.Sequential(*layers)

        self.cls_logits = nn.Conv2d(hc, na * nc, kernel_size=3, padding=1)
        self.box_pred = nn.Conv2d(hc, na * 4, kernel_size=3, padding=1)
        self.mask_coeff = nn.Conv2d(hc, na * np, kernel_size=3, padding=1)

        nn.init.normal_(self.cls_logits.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.cls_logits.bias, 0.0)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        t = self.tower(x)
        return self.cls_logits(t), self.box_pred(t), self.mask_coeff(t)


class YOLACTLite(nn.Module):
    """YOLACT-style one-stage instance segmentation (toy-first, pure torch).

    This is a compact educational implementation: it outputs raw predictions:
    - proto: (B, P, H/4, W/4)
    - cls_logits: (B, A*C, H/8, W/8)
    - bbox_deltas: (B, A*4, H/8, W/8)
    - mask_coeffs: (B, A*P, H/8, W/8)
    """

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        num_protos: int = 32,
        num_anchors: int = 3,
        stem_channels: int = 24,
        low_channels: int = 48,
        det_channels: int = 96,
        backbone_depth: int = 2,
        head_channels: int = 96,
        head_convs: int = 2,
        proto_depth: int = 3,
    ) -> None:
        super().__init__()
        self.stride_proto = 4
        self.stride_det = 8

        self.backbone = _BackboneLowDet(
            in_channels=int(in_channels),
            stem_channels=int(stem_channels),
            low_channels=int(low_channels),
            det_channels=int(det_channels),
            depth=int(backbone_depth),
        )
        self.proto_net = _ProtoNet(int(low_channels), int(num_protos), depth=int(proto_depth))
        self.pred_head = _PredictionHead(
            in_ch=int(det_channels),
            num_classes=int(num_classes),
            num_anchors=int(num_anchors),
            num_protos=int(num_protos),
            head_channels=int(head_channels),
            num_convs=int(head_convs),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = x.to(torch.float32)
        if x.ndim != 4:
            raise ValueError(f"Expected input shape (B, C, H, W), got {tuple(x.shape)}")

        low, det = self.backbone(x)
        proto = self.proto_net(low)
        cls_logits, bbox_deltas, mask_coeffs = self.pred_head(det)
        return {
            "proto": proto,
            "cls_logits": cls_logits,
            "bbox_deltas": bbox_deltas,
            "mask_coeffs": mask_coeffs,
        }


_VARIANTS: dict[str, dict] = {
    "yolact_tiny": {"stem": 24, "low": 40, "det": 80, "depth": 1, "head": 80, "protos": 16},
    "yolact_small": {"stem": 24, "low": 48, "det": 96, "depth": 2, "head": 96, "protos": 32},
    "yolact_base": {"stem": 32, "low": 64, "det": 128, "depth": 3, "head": 128, "protos": 48},
}


def build_yolact_instance_segmenter(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "yolact_small",
    width_mult: float = 1.0,
    num_anchors: int = 3,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown YOLACT variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]

    stem = scale_channels(int(spec["stem"]), float(width_mult), min_ch=16, divisor=8)
    low = scale_channels(int(spec["low"]), float(width_mult), min_ch=16, divisor=8)
    det = scale_channels(int(spec["det"]), float(width_mult), min_ch=16, divisor=8)
    head = scale_channels(int(spec["head"]), float(width_mult), min_ch=16, divisor=8)
    protos = max(8, int(round(int(spec["protos"]) * float(width_mult))))

    return YOLACTLite(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        num_protos=int(protos),
        num_anchors=int(num_anchors),
        stem_channels=int(stem),
        low_channels=int(low),
        det_channels=int(det),
        backbone_depth=int(spec["depth"]),
        head_channels=int(head),
        head_convs=2,
        proto_depth=3,
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_yolact_instance_segmenter(in_channels=3, num_classes=3, variant="yolact_tiny")
    out = m(x)
    print("yolact_tiny", {k: tuple(v.shape) for k, v in out.items()})
    loss = sum(v.mean() for v in out.values())
    loss.backward()
    print("ok")
