import torch
import torch.nn.functional as F
from torch import nn

from dlhub.vision.backbones._blocks import ConvBNAct, scale_channels
from dlhub.vision.detection._detr_utils import MLP, SimpleTransformer, flatten_hw
from dlhub.vision.panoptic_segmentation._common import (
    FPN4,
    BackboneC2C3C4C5,
    check_nchw,
    fuse_panoptic,
)


class DeformableDETRPanoptic(nn.Module):
    """Deformable-DETR-style panoptic segmentation (toy-first).

    This toy version mimics the key idea: multi-scale memory tokens and query-based masks.
    It does NOT implement true deformable attention; instead it uses a lightweight transformer
    over concatenated multi-scale tokens.
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
        num_heads: int = 4,
        num_queries: int = 50,
        enc_layers: int = 1,
        dec_layers: int = 2,
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
        q = int(num_queries)
        if q <= 0:
            raise ValueError("num_queries must be > 0")

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

        self.pix_proj = nn.Conv2d(int(fpn_channels), dm, kernel_size=1, bias=True)  # p2 pixel map
        self.m3 = nn.Conv2d(int(fpn_channels), dm, kernel_size=1, bias=True)
        self.m4 = nn.Conv2d(int(fpn_channels), dm, kernel_size=1, bias=True)
        self.m5 = nn.Conv2d(int(fpn_channels), dm, kernel_size=1, bias=True)

        self.semantic_head = nn.Sequential(
            ConvBNAct(dm, dm, kernel_size=3, stride=1, act="relu"),
            nn.Conv2d(dm, nt + ns, kernel_size=1, bias=True),
        )

        self.transformer = SimpleTransformer(
            dim=dm,
            num_heads=int(num_heads),
            num_encoder_layers=int(enc_layers),
            num_decoder_layers=int(dec_layers),
            mlp_ratio=4.0,
            dropout=0.0,
        )
        self.query_embed = nn.Parameter(torch.randn(q, dm) * 0.02)
        self.cls = nn.Linear(dm, nt)
        self.box = MLP(dm, dm, 4, num_layers=3, act="relu")
        self.mask_embed = nn.Linear(dm, dm)

        self.num_thing_classes = nt
        self.num_stuff_classes = ns

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = check_nchw(x)
        b, _, h, w = x.shape
        c2, c3, c4, c5 = self.backbone(x)
        p2, p3, p4, p5 = self.fpn(c2, c3, c4, c5)

        pix = self.pix_proj(p2)  # (B,DM,H/4,W/4)
        semantic_logits = self.semantic_head(pix)
        semantic_logits = F.interpolate(semantic_logits, size=(h, w), mode="nearest")

        m3 = self.m3(p3)
        m4 = self.m4(p4)
        m5 = self.m5(p5)
        memory = torch.cat([flatten_hw(m3), flatten_hw(m4), flatten_hw(m5)], dim=1)
        queries = self.query_embed.unsqueeze(0).expand(b, -1, -1)
        hs = self.transformer(memory, queries)

        query_cls_logits = self.cls(hs)
        query_boxes = torch.sigmoid(self.box(hs))
        me = self.mask_embed(hs)
        pix_flat = pix.flatten(2)  # (B,DM,HW)
        mask_flat = torch.bmm(me, pix_flat)
        mask_logits = mask_flat.view(b, me.shape[1], pix.shape[-2], pix.shape[-1])
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
    "deformable_detr_panoptic_tiny": {
        "stem": 24,
        "c2": 40,
        "c3": 48,
        "c4": 64,
        "c5": 80,
        "depth": 1,
        "fpn": 96,
        "d_model": 96,
        "heads": 4,
        "q": 25,
        "enc": 1,
        "dec": 1,
    },
    "deformable_detr_panoptic_small": {
        "stem": 32,
        "c2": 48,
        "c3": 64,
        "c4": 96,
        "c5": 128,
        "depth": 2,
        "fpn": 128,
        "d_model": 128,
        "heads": 4,
        "q": 50,
        "enc": 1,
        "dec": 2,
    },
    "deformable_detr_panoptic_base": {
        "stem": 48,
        "c2": 64,
        "c3": 96,
        "c4": 144,
        "c5": 192,
        "depth": 3,
        "fpn": 192,
        "d_model": 192,
        "heads": 6,
        "q": 100,
        "enc": 2,
        "dec": 3,
    },
}


def build_deformable_detr_panoptic_segmenter(
    *,
    in_channels: int,
    num_thing_classes: int,
    num_stuff_classes: int,
    variant: str = "deformable_detr_panoptic_small",
    width_mult: float = 1.0,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(
            f"Unknown Deformable-DETR-panoptic variant: {variant!r}. Supported: {sorted(_VARIANTS)}"
        )
    spec = _VARIANTS[name]

    stem = scale_channels(int(spec["stem"]), float(width_mult), min_ch=16, divisor=8)
    c2 = scale_channels(int(spec["c2"]), float(width_mult), min_ch=16, divisor=8)
    c3 = scale_channels(int(spec["c3"]), float(width_mult), min_ch=16, divisor=8)
    c4 = scale_channels(int(spec["c4"]), float(width_mult), min_ch=16, divisor=8)
    c5 = scale_channels(int(spec["c5"]), float(width_mult), min_ch=16, divisor=8)
    fpn = scale_channels(int(spec["fpn"]), float(width_mult), min_ch=32, divisor=8)
    dm = scale_channels(int(spec["d_model"]), float(width_mult), min_ch=32, divisor=8)
    heads = int(spec["heads"])
    while heads > 1 and dm % heads != 0:
        heads -= 1
    q = max(4, int(round(int(spec["q"]) * float(width_mult))))

    return DeformableDETRPanoptic(
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
        num_heads=int(heads),
        num_queries=int(q),
        enc_layers=int(spec["enc"]),
        dec_layers=int(spec["dec"]),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_deformable_detr_panoptic_segmenter(
        in_channels=3,
        num_thing_classes=3,
        num_stuff_classes=2,
        variant="deformable_detr_panoptic_tiny",
        width_mult=0.5,
    )
    out = m(x)
    print("deformable_detr_panoptic_tiny", {k: tuple(v.shape) for k, v in out.items()})
    loss = (
        out["semantic_logits"].mean() + out["query_cls_logits"].mean() + out["mask_logits"].mean()
    )
    loss.backward()
    print("ok")
