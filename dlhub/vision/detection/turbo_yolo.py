import torch
import torch.nn.functional as F
from torch import nn

from dlhub.vision.backbones._blocks import ConvBNAct, scale_channels
from dlhub.vision.detection._common import check_nchw


class C2fBlock(nn.Module):
    """Tiny C2f-style aggregation block used by YOLOv8."""

    def __init__(self, channels: int, *, depth: int = 2) -> None:
        super().__init__()
        c = int(channels)
        d = int(depth)
        if d <= 0:
            raise ValueError("depth must be > 0")
        hidden = max(8, c // 2)
        self.reduce = ConvBNAct(c, hidden * 2, kernel_size=1, stride=1, padding=0, act="silu")
        self.blocks = nn.ModuleList(
            [ConvBNAct(hidden, hidden, kernel_size=3, stride=1, act="silu") for _ in range(d)]
        )
        self.fuse = ConvBNAct(hidden * (d + 2), c, kernel_size=1, stride=1, padding=0, act="silu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.reduce(x)
        y1, y2 = torch.chunk(y, 2, dim=1)
        feats = [y1, y2]
        cur = y2
        for block in self.blocks:
            cur = block(cur)
            feats.append(cur)
        return self.fuse(torch.cat(feats, dim=1))


class YOLOv8Backbone(nn.Module):
    """Toy YOLOv8 backbone returning stride-8/16/32 features."""

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
        stem = int(stem_channels)
        d = int(depth)
        self.stem = nn.Sequential(
            ConvBNAct(int(in_channels), stem, kernel_size=3, stride=2, act="silu"),
            ConvBNAct(stem, stem, kernel_size=3, stride=2, act="silu"),
        )
        self.s3 = nn.Sequential(
            ConvBNAct(stem, int(c3_channels), kernel_size=3, stride=2, act="silu"),
            C2fBlock(int(c3_channels), depth=d),
        )
        self.s4 = nn.Sequential(
            ConvBNAct(int(c3_channels), int(c4_channels), kernel_size=3, stride=2, act="silu"),
            C2fBlock(int(c4_channels), depth=d),
        )
        self.s5 = nn.Sequential(
            ConvBNAct(int(c4_channels), int(c5_channels), kernel_size=3, stride=2, act="silu"),
            C2fBlock(int(c5_channels), depth=d),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        c3 = self.s3(x)
        c4 = self.s4(c3)
        c5 = self.s5(c4)
        return c3, c4, c5


class YOLOv8PAN(nn.Module):
    def __init__(self, in_channels: tuple[int, int, int], out_channels: int, *, depth: int) -> None:
        super().__init__()
        c3, c4, c5 = (int(v) for v in in_channels)
        out = int(out_channels)
        d = int(depth)
        self.l3 = nn.Conv2d(c3, out, kernel_size=1)
        self.l4 = nn.Conv2d(c4, out, kernel_size=1)
        self.l5 = nn.Conv2d(c5, out, kernel_size=1)

        self.top4 = C2fBlock(out, depth=d)
        self.top3 = C2fBlock(out, depth=d)
        self.down4 = ConvBNAct(out, out, kernel_size=3, stride=2, act="silu")
        self.bot4 = C2fBlock(out, depth=d)
        self.down5 = ConvBNAct(out, out, kernel_size=3, stride=2, act="silu")
        self.bot5 = C2fBlock(out, depth=d)

    def forward(
        self, c3: torch.Tensor, c4: torch.Tensor, c5: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        p5 = self.l5(c5)
        p4 = self.top4(self.l4(c4) + F.interpolate(p5, size=c4.shape[-2:], mode="nearest"))
        p3 = self.top3(self.l3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="nearest"))
        p4 = self.bot4(p4 + self.down4(p3))
        p5 = self.bot5(p5 + self.down5(p4))
        return p3, p4, p5


class YOLOv8Head(nn.Module):
    """Anchor-free decoupled head with a tiny DFL-style branch."""

    def __init__(
        self, *, channels: int, num_classes: int, reg_max: int = 8, num_convs: int = 2
    ) -> None:
        super().__init__()
        c = int(channels)
        nc = int(num_classes)
        rm = int(reg_max)
        n = int(num_convs)
        if nc <= 0:
            raise ValueError("num_classes must be > 0")
        if rm <= 1:
            raise ValueError("reg_max must be > 1")
        self.stem = ConvBNAct(c, c, kernel_size=1, stride=1, padding=0, act="silu")
        self.cls_tower = nn.Sequential(
            *[ConvBNAct(c, c, kernel_size=3, stride=1, act="silu") for _ in range(n)]
        )
        self.reg_tower = nn.Sequential(
            *[ConvBNAct(c, c, kernel_size=3, stride=1, act="silu") for _ in range(n)]
        )
        self.obj = nn.Conv2d(c, 1, kernel_size=1)
        self.cls = nn.Conv2d(c, nc, kernel_size=1)
        self.box = nn.Conv2d(c, 4, kernel_size=1)
        self.dfl = nn.Conv2d(c, 4 * rm, kernel_size=1)

    def forward_single(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        cls_feat = self.cls_tower(x)
        reg_feat = self.reg_tower(x)
        return self.obj(reg_feat), self.cls(cls_feat), self.box(reg_feat), self.dfl(reg_feat)


class TurboYoloDetector(nn.Module):
    """YOLOv8-style detector with C2f backbone and anchor-free head."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        stem_channels: int = 32,
        backbone_channels: tuple[int, int, int] = (64, 96, 128),
        backbone_depth: int = 2,
        neck_channels: int = 96,
        neck_depth: int = 1,
        head_convs: int = 2,
        reg_max: int = 8,
    ) -> None:
        super().__init__()
        c3, c4, c5 = (int(v) for v in backbone_channels)
        out = int(neck_channels)
        self.backbone = YOLOv8Backbone(
            in_channels=int(in_channels),
            stem_channels=int(stem_channels),
            c3_channels=c3,
            c4_channels=c4,
            c5_channels=c5,
            depth=int(backbone_depth),
        )
        self.neck = YOLOv8PAN((c3, c4, c5), out, depth=int(neck_depth))
        self.head = YOLOv8Head(
            channels=out,
            num_classes=int(num_classes),
            reg_max=int(reg_max),
            num_convs=int(head_convs),
        )

    def forward(self, x: torch.Tensor) -> dict[str, list[torch.Tensor]]:
        x = check_nchw(x)
        c3, c4, c5 = self.backbone(x)
        p3, p4, p5 = self.neck(c3, c4, c5)
        obj, cls, box, dfl = [], [], [], []
        for p in (p3, p4, p5):
            o, c, b, d = self.head.forward_single(p)
            obj.append(o)
            cls.append(c)
            box.append(b)
            dfl.append(d)
        return {"obj_logits": obj, "cls_logits": cls, "bbox_deltas": box, "dfl_logits": dfl}


_VARIANTS: dict[str, dict[str, int]] = {
    "turbo_yolo_tiny": {
        "stem": 24,
        "c3": 48,
        "c4": 64,
        "c5": 80,
        "backbone_depth": 1,
        "neck": 64,
        "neck_depth": 1,
        "head": 1,
        "reg_max": 8,
    },
    "turbo_yolo_small": {
        "stem": 32,
        "c3": 64,
        "c4": 96,
        "c5": 128,
        "backbone_depth": 2,
        "neck": 96,
        "neck_depth": 1,
        "head": 2,
        "reg_max": 8,
    },
    "turbo_yolo_base": {
        "stem": 48,
        "c3": 96,
        "c4": 144,
        "c5": 192,
        "backbone_depth": 3,
        "neck": 128,
        "neck_depth": 2,
        "head": 2,
        "reg_max": 12,
    },
}


def build_turbo_yolo_detector(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "turbo_yolo_tiny",
    width_mult: float = 1.0,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown YOLOv8 variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    stem = scale_channels(int(spec["stem"]), float(width_mult), min_ch=16, divisor=8)
    c3 = scale_channels(int(spec["c3"]), float(width_mult), min_ch=16, divisor=8)
    c4 = scale_channels(int(spec["c4"]), float(width_mult), min_ch=16, divisor=8)
    c5 = scale_channels(int(spec["c5"]), float(width_mult), min_ch=16, divisor=8)
    neck = scale_channels(int(spec["neck"]), float(width_mult), min_ch=16, divisor=8)
    return TurboYoloDetector(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        stem_channels=int(stem),
        backbone_channels=(int(c3), int(c4), int(c5)),
        backbone_depth=int(spec["backbone_depth"]),
        neck_channels=int(neck),
        neck_depth=int(spec["neck_depth"]),
        head_convs=int(spec["head"]),
        reg_max=int(spec["reg_max"]),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 128, 128)
    m = build_turbo_yolo_detector(
        in_channels=3, num_classes=3, variant="turbo_yolo_tiny", width_mult=0.5
    )
    out = m(x)
    print("turbo_yolo_tiny", [tuple(t.shape) for t in out["obj_logits"]])
    loss = (
        sum(t.mean() for t in out["obj_logits"])
        + sum(t.mean() for t in out["cls_logits"])
        + sum(t.mean() for t in out["bbox_deltas"])
        + sum(t.mean() for t in out["dfl_logits"])
    )
    loss.backward()
    print("ok")
