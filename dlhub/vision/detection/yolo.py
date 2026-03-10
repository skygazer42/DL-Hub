import torch
from torch import nn

from dlhub.vision.backbones._blocks import ConvBNAct, scale_channels


class _TinyBackbone(nn.Module):
    """A tiny conv backbone that outputs a stride-4 feature map."""

    def __init__(
        self,
        *,
        in_channels: int,
        stem_channels: int,
        out_channels: int,
        depth: int,
    ) -> None:
        super().__init__()
        c_in = int(in_channels)
        stem = int(stem_channels)
        out = int(out_channels)
        d = int(depth)
        if d <= 0:
            raise ValueError("depth must be > 0")

        layers: list[nn.Module] = [
            ConvBNAct(c_in, stem, kernel_size=3, stride=2, act="relu"),
            ConvBNAct(stem, out, kernel_size=3, stride=2, act="relu"),
        ]
        for _ in range(d):
            layers.append(ConvBNAct(out, out, kernel_size=3, stride=1, act="relu"))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class YOLOv1Detector(nn.Module):
    """YOLOv1-style grid detector (toy-first, single object per image).

    Output (raw, stride=4):
    - obj_logits: (B, 1, H/4, W/4)
    - cls_logits: (B, C, H/4, W/4)  (for toy we keep C small; can be 1)
    - bbox: (B, 4, H/4, W/4)  (normalized cx, cy, w, h in [0,1] via sigmoid)

    Notes:
    - We intentionally keep it single-level and do not implement NMS/decoding here.
    - Training code is in the lesson (toy-first).
    """

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        stem_channels: int = 32,
        hidden_channels: int = 64,
        backbone_depth: int = 2,
        head_channels: int = 64,
    ) -> None:
        super().__init__()
        c_in = int(in_channels)
        nc = int(num_classes)
        if nc <= 0:
            raise ValueError("num_classes must be > 0")

        self.stride = 4
        self.num_classes = nc
        self.backbone = _TinyBackbone(
            in_channels=c_in,
            stem_channels=int(stem_channels),
            out_channels=int(hidden_channels),
            depth=int(backbone_depth),
        )

        hc = int(head_channels)
        self.head = nn.Sequential(
            ConvBNAct(int(hidden_channels), hc, kernel_size=3, stride=1, act="relu"),
            nn.Conv2d(hc, 1 + nc + 4, kernel_size=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = x.to(torch.float32)
        if x.ndim != 4:
            raise ValueError(f"Expected input shape (B, C, H, W), got {tuple(x.shape)}")

        feats = self.backbone(x)
        pred = self.head(feats)  # (B, 1+C+4, Gh, Gw)
        obj = pred[:, :1]
        cls = pred[:, 1 : 1 + int(self.num_classes)]
        bbox_raw = pred[:, -4:]
        bbox = torch.sigmoid(bbox_raw)
        return {"obj_logits": obj, "cls_logits": cls, "bbox": bbox}


_VARIANTS: dict[str, dict] = {
    "yolo_v1_tiny": {"stem": 24, "hidden": 32, "depth": 1, "head": 32},
    "yolo_v1_small": {"stem": 32, "hidden": 64, "depth": 2, "head": 64},
    "yolo_v1_base": {"stem": 48, "hidden": 96, "depth": 3, "head": 96},
}


def build_yolo_v1_detector(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "yolo_v1_tiny",
    width_mult: float = 1.0,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown YOLOv1 variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    stem = scale_channels(int(spec["stem"]), float(width_mult), min_ch=16, divisor=8)
    hidden = scale_channels(int(spec["hidden"]), float(width_mult), min_ch=16, divisor=8)
    head = scale_channels(int(spec["head"]), float(width_mult), min_ch=16, divisor=8)
    return YOLOv1Detector(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        stem_channels=int(stem),
        hidden_channels=int(hidden),
        backbone_depth=int(spec["depth"]),
        head_channels=int(head),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_yolo_v1_detector(in_channels=3, num_classes=2, variant="yolo_v1_tiny", width_mult=1.0)
    out = m(x)
    print("yolo_v1_tiny", {k: tuple(v.shape) for k, v in out.items()})
    loss = sum(v.mean() for v in out.values())
    loss.backward()
    print("ok")
