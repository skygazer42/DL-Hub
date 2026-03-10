import torch
from torch import nn

from dlhub.vision.detection.yolov8 import build_yolov8_detector as _build_base

_VARIANTS: dict[str, str] = {
    "gold_yolo_tiny": "yolov8_tiny",
    "gold_yolo_small": "yolov8_small",
    "gold_yolo_base": "yolov8_base",
}


def build_gold_yolo_detector(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "gold_yolo_tiny",
    width_mult: float = 1.0,
) -> nn.Module:
    name = str(variant).lower().strip()
    base_variant = _VARIANTS.get(name)
    if base_variant is None:
        raise ValueError(f"Unknown Gold-YOLO variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    return _build_base(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        variant=base_variant,
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 128, 128)
    model = build_gold_yolo_detector(
        in_channels=3, num_classes=3, variant="gold_yolo_tiny", width_mult=0.5
    )
    out = model(x)
    print("gold_yolo_tiny", [tuple(t.shape) for t in out["cls_logits"]])
    loss = sum(t.mean() for v in out.values() for t in (v if isinstance(v, list) else [v]))
    loss.backward()
    print("ok")
