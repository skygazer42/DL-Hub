import torch
import torch.nn.functional as F
from torch import nn

from dlhub.vision.backbones._blocks import ConvBNAct, scale_channels
from dlhub.vision.panoptic_segmentation._common import (
    FPN4,
    BackboneC2C3C4C5,
    check_nchw,
    fuse_panoptic,
)


class SparseRCNNPanoptic(nn.Module):
    """Sparse R-CNN-style panoptic segmentation (toy-first).

    Uses learnable proposal features + proposal boxes refined by a few MLP layers.
    Masks are produced by dot-product against a high-res pixel embedding map.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        num_thing_classes: int,
        num_stuff_classes: int,
        stem_channels: int = 32,
        c2_channels: int = 48,
        c3_channels: int = 64,
        c4_channels: int = 96,
        c5_channels: int = 128,
        depth: int = 2,
        fpn_channels: int = 128,
        d_model: int = 128,
        num_proposals: int = 50,
        refine_layers: int = 3,
        mlp_ratio: float = 2.0,
    ) -> None:
        super().__init__()
        nt = int(num_thing_classes)
        ns = int(num_stuff_classes)
        if nt <= 0:
            raise ValueError("num_thing_classes must be > 0")
        if ns <= 0:
            raise ValueError("num_stuff_classes must be > 0")
        dm = int(d_model)
        if dm <= 0:
            raise ValueError("d_model must be > 0")
        np = int(num_proposals)
        if np <= 0:
            raise ValueError("num_proposals must be > 0")
        rl = int(refine_layers)
        if rl <= 0:
            raise ValueError("refine_layers must be > 0")

        self.backbone = BackboneC2C3C4C5(
            in_channels=int(in_channels),
            stem_channels=int(stem_channels),
            c2_channels=int(c2_channels),
            c3_channels=int(c3_channels),
            c4_channels=int(c4_channels),
            c5_channels=int(c5_channels),
            depth=int(depth),
            act="relu",
        )
        self.fpn = FPN4(
            (int(c2_channels), int(c3_channels), int(c4_channels), int(c5_channels)),
            int(fpn_channels),
            act="relu",
        )

        self.pix_proj = nn.Conv2d(int(fpn_channels), dm, kernel_size=1, bias=True)
        self.semantic_head = nn.Sequential(
            ConvBNAct(dm, dm, kernel_size=3, stride=1, act="relu"),
            nn.Conv2d(dm, nt + ns, kernel_size=1, bias=True),
        )

        self.proposals = nn.Parameter(torch.randn(np, dm) * 0.02)
        self.proposal_boxes = nn.Parameter(torch.rand(np, 4))

        hidden = max(8, int(round(dm * float(mlp_ratio))))
        self.refine = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(dm),
                    nn.Linear(dm, hidden),
                    nn.ReLU(inplace=True),
                    nn.Linear(hidden, dm),
                )
                for _ in range(rl)
            ]
        )

        self.cls = nn.Linear(dm, nt)
        self.box = nn.Linear(dm, 4)
        self.mask_embed = nn.Linear(dm, dm)

        self.num_proposals = np
        self.num_thing_classes = nt
        self.num_stuff_classes = ns

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = check_nchw(x)
        b, _, h, w = x.shape
        c2, c3, c4, c5 = self.backbone(x)
        p2, _, _, p5 = self.fpn(c2, c3, c4, c5)

        pix = self.pix_proj(p2)
        semantic_logits = self.semantic_head(pix)
        semantic_logits = F.interpolate(semantic_logits, size=(h, w), mode="nearest")

        ctx = F.adaptive_avg_pool2d(p5, (1, 1)).flatten(1).unsqueeze(1)  # (B,1,C)
        prop = self.proposals.unsqueeze(0).expand(b, -1, -1) + ctx
        for layer in self.refine:
            prop = prop + layer(prop)

        query_cls_logits = self.cls(prop)
        query_boxes = torch.sigmoid(self.box(prop) + self.proposal_boxes.sigmoid().unsqueeze(0))
        me = self.mask_embed(prop)
        mask_flat = torch.bmm(me, pix.flatten(2))
        mask_logits = mask_flat.view(b, self.num_proposals, pix.shape[-2], pix.shape[-1])
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
            "panoptic_map": panoptic_map,
        }


_VARIANTS: dict[str, dict] = {
    "sparse_rcnn_panoptic_tiny": {
        "stem": 24,
        "c2": 40,
        "c3": 48,
        "c4": 64,
        "c5": 80,
        "depth": 1,
        "fpn": 96,
        "d_model": 96,
        "props": 25,
        "refine": 2,
    },
    "sparse_rcnn_panoptic_small": {
        "stem": 32,
        "c2": 48,
        "c3": 64,
        "c4": 96,
        "c5": 128,
        "depth": 2,
        "fpn": 128,
        "d_model": 128,
        "props": 50,
        "refine": 3,
    },
    "sparse_rcnn_panoptic_base": {
        "stem": 48,
        "c2": 64,
        "c3": 96,
        "c4": 144,
        "c5": 192,
        "depth": 3,
        "fpn": 192,
        "d_model": 192,
        "props": 100,
        "refine": 4,
    },
}


def build_sparse_rcnn_panoptic_segmenter(
    *,
    in_channels: int,
    num_thing_classes: int,
    num_stuff_classes: int,
    variant: str = "sparse_rcnn_panoptic_small",
    width_mult: float = 1.0,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(
            f"Unknown Sparse R-CNN-panoptic variant: {variant!r}. Supported: {sorted(_VARIANTS)}"
        )
    spec = _VARIANTS[name]

    stem = scale_channels(int(spec["stem"]), float(width_mult), min_ch=16, divisor=8)
    c2 = scale_channels(int(spec["c2"]), float(width_mult), min_ch=16, divisor=8)
    c3 = scale_channels(int(spec["c3"]), float(width_mult), min_ch=16, divisor=8)
    c4 = scale_channels(int(spec["c4"]), float(width_mult), min_ch=16, divisor=8)
    c5 = scale_channels(int(spec["c5"]), float(width_mult), min_ch=16, divisor=8)
    fpn = scale_channels(int(spec["fpn"]), float(width_mult), min_ch=32, divisor=8)
    dm = scale_channels(int(spec["d_model"]), float(width_mult), min_ch=32, divisor=8)
    props = max(4, int(round(int(spec["props"]) * float(width_mult))))

    return SparseRCNNPanoptic(
        in_channels=int(in_channels),
        num_thing_classes=int(num_thing_classes),
        num_stuff_classes=int(num_stuff_classes),
        stem_channels=int(stem),
        c2_channels=int(c2),
        c3_channels=int(c3),
        c4_channels=int(c4),
        c5_channels=int(c5),
        depth=int(spec["depth"]),
        fpn_channels=int(fpn),
        d_model=int(dm),
        num_proposals=int(props),
        refine_layers=int(spec["refine"]),
        mlp_ratio=2.0,
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_sparse_rcnn_panoptic_segmenter(
        in_channels=3,
        num_thing_classes=3,
        num_stuff_classes=2,
        variant="sparse_rcnn_panoptic_tiny",
        width_mult=0.5,
    )
    out = m(x)
    print("sparse_rcnn_panoptic_tiny", {k: tuple(v.shape) for k, v in out.items()})
    loss = (
        out["semantic_logits"].mean() + out["query_cls_logits"].mean() + out["mask_logits"].mean()
    )
    loss.backward()
    print("ok")
