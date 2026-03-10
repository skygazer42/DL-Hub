import torch
import torch.nn.functional as F
from torch import nn

from dlhub.vision.backbones._blocks import ConvBNAct, DepthwiseSeparableConv, scale_channels
from dlhub.vision.detection._detr_utils import MLP, SimpleTransformer, flatten_hw
from dlhub.vision.panoptic_segmentation._common import BackboneC2C3C4C5, check_nchw, fuse_panoptic


class RTDETRPanoptic(nn.Module):
    """RT-DETR-style panoptic segmentation (toy-first).

    Uses multi-scale conv features -> lightweight conv encoder -> transformer queries.
    Masks are produced via dot-product against a pixel embedding map.
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
        d_model: int = 128,
        num_heads: int = 4,
        num_queries: int = 50,
        enc_layers: int = 1,
        dec_layers: int = 2,
        conv_encoder_layers: int = 2,
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
            act="silu",
        )

        self.p2 = nn.Conv2d(int(c2_channels), dm, kernel_size=1)
        self.p3 = nn.Conv2d(int(c3_channels), dm, kernel_size=1)
        self.p4 = nn.Conv2d(int(c4_channels), dm, kernel_size=1)
        self.p5 = nn.Conv2d(int(c5_channels), dm, kernel_size=1)

        enc: list[nn.Module] = []
        for _ in range(int(conv_encoder_layers)):
            enc.append(DepthwiseSeparableConv(dm, dm, act="silu"))
        self.conv_encoder = nn.Sequential(*enc)

        self.semantic_head = nn.Sequential(
            ConvBNAct(dm, dm, kernel_size=3, stride=1, act="silu"),
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
        self.class_head = nn.Linear(dm, nt)
        self.box_head = MLP(dm, dm, 4, num_layers=3, act="relu")
        self.mask_embed = nn.Linear(dm, dm)

        self.num_thing_classes = nt
        self.num_stuff_classes = ns

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = check_nchw(x)
        b, _, h, w = x.shape
        c2, c3, c4, c5 = self.backbone(x)
        p2 = self.conv_encoder(self.p2(c2))
        p3 = self.conv_encoder(self.p3(c3))
        p4 = self.conv_encoder(self.p4(c4))
        p5 = self.conv_encoder(self.p5(c5))

        semantic_logits = self.semantic_head(p2)
        semantic_logits = F.interpolate(semantic_logits, size=(h, w), mode="nearest")

        memory = torch.cat([flatten_hw(p3), flatten_hw(p4), flatten_hw(p5)], dim=1)
        queries = self.query_embed.unsqueeze(0).expand(b, -1, -1).contiguous()
        hs = self.transformer(memory, queries)  # (B,Q,DM)

        query_cls_logits = self.class_head(hs)
        query_boxes = torch.sigmoid(self.box_head(hs))
        me = self.mask_embed(hs)
        pix_flat = p2.flatten(2)  # (B,DM,H2W2)
        mask_flat = torch.bmm(me, pix_flat)
        mask_logits = mask_flat.view(b, me.shape[1], p2.shape[-2], p2.shape[-1])
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
    "rtdetr_panoptic_tiny": {
        "stem": 24,
        "c2": 40,
        "c3": 48,
        "c4": 64,
        "c5": 80,
        "depth": 1,
        "d_model": 96,
        "heads": 4,
        "q": 25,
        "enc": 1,
        "dec": 1,
        "conv": 1,
    },
    "rtdetr_panoptic_small": {
        "stem": 32,
        "c2": 48,
        "c3": 64,
        "c4": 96,
        "c5": 128,
        "depth": 2,
        "d_model": 128,
        "heads": 4,
        "q": 50,
        "enc": 1,
        "dec": 2,
        "conv": 2,
    },
    "rtdetr_panoptic_base": {
        "stem": 48,
        "c2": 64,
        "c3": 96,
        "c4": 144,
        "c5": 192,
        "depth": 3,
        "d_model": 192,
        "heads": 6,
        "q": 100,
        "enc": 2,
        "dec": 3,
        "conv": 2,
    },
}


def build_rtdetr_panoptic_segmenter(
    *,
    in_channels: int,
    num_thing_classes: int,
    num_stuff_classes: int,
    variant: str = "rtdetr_panoptic_small",
    width_mult: float = 1.0,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(
            f"Unknown RT-DETR-panoptic variant: {variant!r}. Supported: {sorted(_VARIANTS)}"
        )
    spec = _VARIANTS[name]

    stem = scale_channels(int(spec["stem"]), float(width_mult), min_ch=16, divisor=8)
    c2 = scale_channels(int(spec["c2"]), float(width_mult), min_ch=16, divisor=8)
    c3 = scale_channels(int(spec["c3"]), float(width_mult), min_ch=16, divisor=8)
    c4 = scale_channels(int(spec["c4"]), float(width_mult), min_ch=16, divisor=8)
    c5 = scale_channels(int(spec["c5"]), float(width_mult), min_ch=16, divisor=8)
    d_model = scale_channels(int(spec["d_model"]), float(width_mult), min_ch=32, divisor=8)
    q = max(4, int(round(int(spec["q"]) * float(width_mult))))

    return RTDETRPanoptic(
        in_channels=int(in_channels),
        num_thing_classes=int(num_thing_classes),
        num_stuff_classes=int(num_stuff_classes),
        stem_channels=int(stem),
        c2_channels=int(c2),
        c3_channels=int(c3),
        c4_channels=int(c4),
        c5_channels=int(c5),
        depth=int(spec["depth"]),
        d_model=int(d_model),
        num_heads=int(spec["heads"]),
        num_queries=int(q),
        enc_layers=int(spec["enc"]),
        dec_layers=int(spec["dec"]),
        conv_encoder_layers=int(spec["conv"]),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_rtdetr_panoptic_segmenter(
        in_channels=3,
        num_thing_classes=3,
        num_stuff_classes=2,
        variant="rtdetr_panoptic_tiny",
        width_mult=0.5,
    )
    out = m(x)
    print("rtdetr_panoptic_tiny", {k: tuple(v.shape) for k, v in out.items()})
    loss = (
        out["semantic_logits"].mean() + out["query_cls_logits"].mean() + out["mask_logits"].mean()
    )
    loss.backward()
    print("ok")
