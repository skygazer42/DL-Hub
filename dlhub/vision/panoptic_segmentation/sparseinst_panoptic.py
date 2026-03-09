
import torch
from torch import nn
import torch.nn.functional as F

from dlhub.vision.backbones._blocks import ConvBNAct, scale_channels
from dlhub.vision.panoptic_segmentation._common import BackboneLowDet, ProtoNet, check_nchw, fuse_panoptic, masks_from_prototypes


class SparseInstPanoptic(nn.Module):
    """SparseInst-style panoptic segmentation (toy-first).

    Learnable instance queries predict class/box and mask coefficients over prototypes, plus a semantic head.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        num_thing_classes: int,
        num_stuff_classes: int,
        num_instances: int = 32,
        num_protos: int = 32,
        stem_channels: int = 24,
        low_channels: int = 48,
        det_channels: int = 96,
        backbone_depth: int = 2,
        query_dim: int = 96,
        proto_depth: int = 3,
    ) -> None:
        super().__init__()
        nt = int(num_thing_classes)
        ns = int(num_stuff_classes)
        if nt <= 0:
            raise ValueError("num_thing_classes must be > 0")
        if ns <= 0:
            raise ValueError("num_stuff_classes must be > 0")
        n = int(num_instances)
        if n <= 0:
            raise ValueError("num_instances must be > 0")
        np = int(num_protos)
        if np <= 0:
            raise ValueError("num_protos must be > 0")
        qd = int(query_dim)
        if qd <= 0:
            raise ValueError("query_dim must be > 0")

        self.backbone = BackboneLowDet(
            in_channels=int(in_channels),
            stem_channels=int(stem_channels),
            low_channels=int(low_channels),
            det_channels=int(det_channels),
            depth=int(backbone_depth),
            act="relu",
        )
        self.semantic = nn.Sequential(
            ConvBNAct(int(low_channels), int(low_channels), kernel_size=3, stride=1, act="relu"),
            nn.Conv2d(int(low_channels), nt + ns, kernel_size=1, bias=True),
        )
        self.proto = ProtoNet(int(low_channels), np, depth=int(proto_depth), act="relu")

        self.query = nn.Parameter(torch.randn(n, qd) * 0.02)
        self.proj = nn.Sequential(nn.Linear(int(det_channels), qd), nn.ReLU(inplace=True))
        self.cls = nn.Linear(qd, nt)
        self.box = nn.Linear(qd, 4)
        self.coeff = nn.Linear(qd, np)

        self.num_thing_classes = nt
        self.num_stuff_classes = ns
        self.num_instances = n

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = check_nchw(x)
        b, _, h, w = x.shape
        low, det = self.backbone(x)

        semantic_logits = self.semantic(low)
        semantic_logits = F.interpolate(semantic_logits, size=(h, w), mode="nearest")

        proto = self.proto(low)  # (B,P,H/4,W/4)

        pooled = F.adaptive_avg_pool2d(det, (1, 1)).flatten(1)
        base = self.proj(pooled).unsqueeze(1)  # (B,1,QD)
        q = self.query.unsqueeze(0).expand(b, -1, -1)
        hq = base + q
        query_cls_logits = self.cls(hq)
        query_boxes = torch.sigmoid(self.box(hq))
        coeff = self.coeff(hq)

        mask_logits = masks_from_prototypes(proto, coeff)
        mask_logits = F.interpolate(mask_logits, size=(h, w), mode="nearest")

        scores = query_cls_logits.softmax(dim=-1).max(dim=-1).values
        panoptic_map = fuse_panoptic(semantic_logits, mask_logits, scores, thing_offset=int(self.num_stuff_classes))

        return {
            "semantic_logits": semantic_logits,
            "query_cls_logits": query_cls_logits,
            "query_boxes": query_boxes,
            "mask_logits": mask_logits,
            "mask_coeff": coeff,
            "panoptic_map": panoptic_map,
        }


_VARIANTS: dict[str, dict] = {
    "sparseinst_panoptic_tiny": {"stem": 24, "low": 40, "det": 80, "depth": 1, "instances": 16, "protos": 16, "qdim": 80},
    "sparseinst_panoptic_small": {"stem": 24, "low": 48, "det": 96, "depth": 2, "instances": 32, "protos": 32, "qdim": 96},
    "sparseinst_panoptic_base": {"stem": 32, "low": 64, "det": 128, "depth": 3, "instances": 64, "protos": 48, "qdim": 128},
}


def build_sparseinst_panoptic_segmenter(
    *,
    in_channels: int,
    num_thing_classes: int,
    num_stuff_classes: int,
    variant: str = "sparseinst_panoptic_small",
    width_mult: float = 1.0,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown SparseInst-panoptic variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]

    stem = scale_channels(int(spec["stem"]), float(width_mult), min_ch=16, divisor=8)
    low = scale_channels(int(spec["low"]), float(width_mult), min_ch=16, divisor=8)
    det = scale_channels(int(spec["det"]), float(width_mult), min_ch=16, divisor=8)
    qdim = scale_channels(int(spec["qdim"]), float(width_mult), min_ch=16, divisor=8)
    protos = max(8, int(round(int(spec["protos"]) * float(width_mult))))
    instances = max(4, int(round(int(spec["instances"]) * float(width_mult))))

    return SparseInstPanoptic(
        in_channels=int(in_channels),
        num_thing_classes=int(num_thing_classes),
        num_stuff_classes=int(num_stuff_classes),
        num_instances=int(instances),
        num_protos=int(protos),
        stem_channels=int(stem),
        low_channels=int(low),
        det_channels=int(det),
        backbone_depth=int(spec["depth"]),
        query_dim=int(qdim),
        proto_depth=3,
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_sparseinst_panoptic_segmenter(
        in_channels=3, num_thing_classes=3, num_stuff_classes=2, variant="sparseinst_panoptic_tiny", width_mult=0.5
    )
    out = m(x)
    print("sparseinst_panoptic_tiny", {k: tuple(v.shape) for k, v in out.items()})
    loss = out["semantic_logits"].mean() + out["query_cls_logits"].mean() + out["mask_logits"].mean()
    loss.backward()
    print("ok")

