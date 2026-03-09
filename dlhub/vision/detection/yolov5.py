
import torch
from torch import nn
import torch.nn.functional as F

from dlhub.vision.backbones._blocks import ConvBNAct, scale_channels
from dlhub.vision.detection._common import check_nchw


class CSPBlock(nn.Module):
    """A tiny CSP-style block (toy)."""

    def __init__(self, channels: int, *, depth: int = 2) -> None:
        super().__init__()
        c = int(channels)
        d = int(depth)
        if d <= 0:
            raise ValueError("depth must be > 0")
        hidden = max(8, c // 2)
        self.cv1 = ConvBNAct(c, hidden, kernel_size=1, stride=1, padding=0, act="silu")
        self.cv2 = ConvBNAct(c, hidden, kernel_size=1, stride=1, padding=0, act="silu")
        blocks: list[nn.Module] = []
        for _ in range(d):
            blocks.append(ConvBNAct(hidden, hidden, kernel_size=3, stride=1, act="silu"))
        self.m = nn.Sequential(*blocks)
        self.cv3 = ConvBNAct(hidden * 2, c, kernel_size=1, stride=1, padding=0, act="silu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y1 = self.m(self.cv1(x))
        y2 = self.cv2(x)
        return self.cv3(torch.cat([y1, y2], dim=1))


class CSPBackboneC3C5(nn.Module):
    """Toy CSPDarknet-ish backbone returning (C3,C4,C5)."""

    def __init__(
        self,
        *,
        in_channels: int,
        stem_channels: int,
        c3_channels: int,
        c4_channels: int,
        c5_channels: int,
        depth: int,
    ) -> None:
        super().__init__()
        c_in = int(in_channels)
        stem = int(stem_channels)
        d = int(depth)
        if d <= 0:
            raise ValueError("depth must be > 0")

        self.stem = nn.Sequential(
            ConvBNAct(c_in, stem, kernel_size=3, stride=2, act="silu"),  # /2
            ConvBNAct(stem, stem, kernel_size=3, stride=2, act="silu"),  # /4
        )

        self.s3 = nn.Sequential(
            ConvBNAct(stem, int(c3_channels), kernel_size=3, stride=2, act="silu"),  # /8
            CSPBlock(int(c3_channels), depth=d),
        )
        self.s4 = nn.Sequential(
            ConvBNAct(int(c3_channels), int(c4_channels), kernel_size=3, stride=2, act="silu"),  # /16
            CSPBlock(int(c4_channels), depth=d),
        )
        self.s5 = nn.Sequential(
            ConvBNAct(int(c4_channels), int(c5_channels), kernel_size=3, stride=2, act="silu"),  # /32
            CSPBlock(int(c5_channels), depth=d),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        c3 = self.s3(x)
        c4 = self.s4(c3)
        c5 = self.s5(c4)
        return c3, c4, c5


class PAFPN(nn.Module):
    def __init__(self, in_channels: tuple[int, int, int], out_channels: int) -> None:
        super().__init__()
        c3, c4, c5 = (int(x) for x in in_channels)
        out = int(out_channels)
        self.l3 = nn.Conv2d(c3, out, kernel_size=1)
        self.l4 = nn.Conv2d(c4, out, kernel_size=1)
        self.l5 = nn.Conv2d(c5, out, kernel_size=1)

        self.s3 = ConvBNAct(out, out, kernel_size=3, stride=1, act="silu")
        self.s4 = ConvBNAct(out, out, kernel_size=3, stride=1, act="silu")
        self.s5 = ConvBNAct(out, out, kernel_size=3, stride=1, act="silu")

        self.down4 = ConvBNAct(out, out, kernel_size=3, stride=2, act="silu")
        self.down5 = ConvBNAct(out, out, kernel_size=3, stride=2, act="silu")

    def forward(self, c3: torch.Tensor, c4: torch.Tensor, c5: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        p5 = self.l5(c5)
        p4 = self.l4(c4) + F.interpolate(p5, size=c4.shape[-2:], mode="nearest")
        p3 = self.l3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="nearest")
        p3, p4, p5 = self.s3(p3), self.s4(p4), self.s5(p5)
        p4 = p4 + self.down4(p3)
        p5 = p5 + self.down5(p4)
        return p3, p4, p5


class YOLOv5Head(nn.Module):
    def __init__(self, *, channels: int, num_classes: int, num_anchors: int = 3) -> None:
        super().__init__()
        c = int(channels)
        nc = int(num_classes)
        na = int(num_anchors)
        if nc <= 0:
            raise ValueError("num_classes must be > 0")
        if na <= 0:
            raise ValueError("num_anchors must be > 0")
        self.obj = nn.Conv2d(c, na, kernel_size=1)
        self.cls = nn.Conv2d(c, na * nc, kernel_size=1)
        self.box = nn.Conv2d(c, na * 4, kernel_size=1)

    def forward_single(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.obj(x), self.cls(x), self.box(x)


class YOLOv5Detector(nn.Module):
    """YOLOv5-style CSP+PAN detector (toy-first)."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        stem_channels: int = 32,
        backbone_channels: tuple[int, int, int] = (64, 96, 128),
        backbone_depth: int = 2,
        fpn_channels: int = 96,
        num_anchors: int = 3,
    ) -> None:
        super().__init__()
        c3, c4, c5 = (int(x) for x in backbone_channels)
        out = int(fpn_channels)
        self.backbone = CSPBackboneC3C5(
            in_channels=int(in_channels),
            stem_channels=int(stem_channels),
            c3_channels=c3,
            c4_channels=c4,
            c5_channels=c5,
            depth=int(backbone_depth),
        )
        self.neck = PAFPN((c3, c4, c5), out)
        self.head = YOLOv5Head(channels=out, num_classes=int(num_classes), num_anchors=int(num_anchors))

    def forward(self, x: torch.Tensor) -> dict[str, list[torch.Tensor]]:
        x = check_nchw(x)
        c3, c4, c5 = self.backbone(x)
        p3, p4, p5 = self.neck(c3, c4, c5)
        obj, cls, box = [], [], []
        for p in (p3, p4, p5):
            o, c, b = self.head.forward_single(p)
            obj.append(o)
            cls.append(c)
            box.append(b)
        return {"obj_logits": obj, "cls_logits": cls, "bbox_deltas": box}


_VARIANTS: dict[str, dict] = {
    "yolov5_tiny": {"stem": 24, "c3": 48, "c4": 64, "c5": 80, "depth": 1, "fpn": 64},
    "yolov5_small": {"stem": 32, "c3": 64, "c4": 96, "c5": 128, "depth": 2, "fpn": 96},
    "yolov5_base": {"stem": 48, "c3": 96, "c4": 144, "c5": 192, "depth": 3, "fpn": 128},
}


def build_yolov5_detector(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "yolov5_tiny",
    width_mult: float = 1.0,
    num_anchors: int = 3,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown YOLOv5 variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    stem = scale_channels(int(spec["stem"]), float(width_mult), min_ch=16, divisor=8)
    c3 = scale_channels(int(spec["c3"]), float(width_mult), min_ch=16, divisor=8)
    c4 = scale_channels(int(spec["c4"]), float(width_mult), min_ch=16, divisor=8)
    c5 = scale_channels(int(spec["c5"]), float(width_mult), min_ch=16, divisor=8)
    fpn = scale_channels(int(spec["fpn"]), float(width_mult), min_ch=16, divisor=8)
    return YOLOv5Detector(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        stem_channels=int(stem),
        backbone_channels=(int(c3), int(c4), int(c5)),
        backbone_depth=int(spec["depth"]),
        fpn_channels=int(fpn),
        num_anchors=int(num_anchors),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 128, 128)
    m = build_yolov5_detector(in_channels=3, num_classes=3, variant="yolov5_tiny", width_mult=0.5)
    out = m(x)
    print("yolov5_tiny", [tuple(t.shape) for t in out["obj_logits"]])
    loss = sum(t.mean() for t in out["obj_logits"]) + sum(t.mean() for t in out["cls_logits"]) + sum(t.mean() for t in out["bbox_deltas"])
    loss.backward()
    print("ok")

