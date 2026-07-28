import torch
import torch.nn.functional as F
from torch import nn

from dlhub.vision.backbones._blocks import ConvBNAct, scale_channels
from dlhub.vision.panoptic_segmentation._common import (
    BackboneLowDet,
    ProtoNet,
    check_nchw,
    fuse_panoptic,
    masks_from_prototypes,
)


class QueryInstPanoptic(nn.Module):
    """QueryInst-style panoptic segmentation (compact-first).

    Learned queries with cross-attention over a feature map; masks via prototypes; plus a semantic head.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        num_thing_classes: int,
        num_stuff_classes: int,
        num_queries: int = 32,
        num_protos: int = 32,
        stem_channels: int = 24,
        low_channels: int = 48,
        det_channels: int = 96,
        backbone_depth: int = 2,
        d_model: int = 96,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        proto_depth: int = 3,
    ) -> None:
        super().__init__()
        nt = int(num_thing_classes)
        ns = int(num_stuff_classes)
        if nt <= 0:
            raise ValueError("num_thing_classes must be > 0")
        if ns <= 0:
            raise ValueError("num_stuff_classes must be > 0")
        nq = int(num_queries)
        if nq <= 0:
            raise ValueError("num_queries must be > 0")
        np = int(num_protos)
        if np <= 0:
            raise ValueError("num_protos must be > 0")
        dm = int(d_model)
        nh = int(num_heads)
        if dm <= 0 or nh <= 0 or dm % nh != 0:
            raise ValueError("d_model must be > 0 and divisible by num_heads")

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

        self.input_proj = nn.Conv2d(int(det_channels), dm, kernel_size=1, bias=True)
        self.query = nn.Parameter(torch.randn(nq, dm) * 0.02)

        self.attn = nn.MultiheadAttention(dm, nh, batch_first=True)
        self.norm1 = nn.LayerNorm(dm)
        self.mlp = nn.Sequential(
            nn.Linear(dm, int(round(dm * float(mlp_ratio)))),
            nn.GELU(),
            nn.Linear(int(round(dm * float(mlp_ratio))), dm),
        )
        self.norm2 = nn.LayerNorm(dm)

        self.cls = nn.Linear(dm, nt)
        self.box = nn.Linear(dm, 4)
        self.coeff = nn.Linear(dm, np)

        self.num_thing_classes = nt
        self.num_stuff_classes = ns

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = check_nchw(x)
        b, _, h, w = x.shape
        low, det = self.backbone(x)

        semantic_logits = self.semantic(low)
        semantic_logits = F.interpolate(semantic_logits, size=(h, w), mode="nearest")

        proto = self.proto(low)  # (B,P,H/4,W/4)

        feat = self.input_proj(det)
        tokens = feat.flatten(2).transpose(1, 2)  # (B,N,DM)
        q = self.query.unsqueeze(0).expand(b, -1, -1)

        attn_out, _ = self.attn(q, tokens, tokens, need_weights=False)
        q = self.norm1(q + attn_out)
        q = self.norm2(q + self.mlp(q))

        query_cls_logits = self.cls(q)
        query_boxes = torch.sigmoid(self.box(q))
        coeff = self.coeff(q)

        mask_logits = masks_from_prototypes(proto, coeff)
        mask_logits = F.interpolate(mask_logits, size=(h, w), mode="nearest")

        scores = query_cls_logits.softmax(dim=-1).max(dim=-1).values
        panoptic_map = fuse_panoptic(
            semantic_logits, mask_logits, scores, thing_offset=int(self.num_stuff_classes)
        )

        return {
            "semantic_logits": semantic_logits,
            "query_cls_logits": query_cls_logits,
            "query_boxes": query_boxes,
            "mask_logits": mask_logits,
            "mask_coeff": coeff,
            "panoptic_map": panoptic_map,
        }


_VARIANTS: dict[str, dict] = {
    "queryinst_panoptic_tiny": {
        "stem": 24,
        "low": 40,
        "det": 80,
        "depth": 1,
        "queries": 16,
        "protos": 16,
        "d_model": 80,
        "heads": 4,
    },
    "queryinst_panoptic_small": {
        "stem": 24,
        "low": 48,
        "det": 96,
        "depth": 2,
        "queries": 32,
        "protos": 32,
        "d_model": 96,
        "heads": 4,
    },
    "queryinst_panoptic_base": {
        "stem": 32,
        "low": 64,
        "det": 128,
        "depth": 3,
        "queries": 64,
        "protos": 48,
        "d_model": 128,
        "heads": 8,
    },
}


def build_queryinst_panoptic_segmenter(
    *,
    in_channels: int,
    num_thing_classes: int,
    num_stuff_classes: int,
    variant: str = "queryinst_panoptic_small",
    width_mult: float = 1.0,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(
            f"Unknown QueryInst-panoptic variant: {variant!r}. Supported: {sorted(_VARIANTS)}"
        )
    spec = _VARIANTS[name]

    stem = scale_channels(int(spec["stem"]), float(width_mult), min_ch=16, divisor=8)
    low = scale_channels(int(spec["low"]), float(width_mult), min_ch=16, divisor=8)
    det = scale_channels(int(spec["det"]), float(width_mult), min_ch=16, divisor=8)
    d_model = scale_channels(int(spec["d_model"]), float(width_mult), min_ch=16, divisor=8)
    heads = int(spec["heads"])
    while heads > 1 and d_model % heads != 0:
        heads -= 1
    protos = max(8, int(round(int(spec["protos"]) * float(width_mult))))
    queries = max(4, int(round(int(spec["queries"]) * float(width_mult))))

    return QueryInstPanoptic(
        in_channels=int(in_channels),
        num_thing_classes=int(num_thing_classes),
        num_stuff_classes=int(num_stuff_classes),
        num_queries=int(queries),
        num_protos=int(protos),
        stem_channels=int(stem),
        low_channels=int(low),
        det_channels=int(det),
        backbone_depth=int(spec["depth"]),
        d_model=int(d_model),
        num_heads=int(heads),
        mlp_ratio=4.0,
        proto_depth=3,
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_queryinst_panoptic_segmenter(
        in_channels=3,
        num_thing_classes=3,
        num_stuff_classes=2,
        variant="queryinst_panoptic_tiny",
        width_mult=0.5,
    )
    out = m(x)
    print("queryinst_panoptic_tiny", {k: tuple(v.shape) for k, v in out.items()})
    loss = (
        out["semantic_logits"].mean() + out["query_cls_logits"].mean() + out["mask_logits"].mean()
    )
    loss.backward()
    print("ok")
