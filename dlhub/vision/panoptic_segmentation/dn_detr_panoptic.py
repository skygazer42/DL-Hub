import torch
import torch.nn.functional as F
from torch import nn

from dlhub.vision.backbones._blocks import scale_channels
from dlhub.vision.panoptic_segmentation._common import BackboneLowDet, check_nchw, fuse_panoptic


class DNDTRPanoptic(nn.Module):
    """DN-DETR-style panoptic segmentation (compact-first).

    Adds an extra set of "denoising" queries alongside the normal DETR queries.
    This is a lightweight educational skeleton (not a full training recipe).
    """

    def __init__(
        self,
        *,
        in_channels: int,
        num_thing_classes: int,
        num_stuff_classes: int,
        num_queries: int = 32,
        num_dn_queries: int = 16,
        stem_channels: int = 24,
        low_channels: int = 48,
        det_channels: int = 96,
        backbone_depth: int = 2,
        d_model: int = 96,
        num_heads: int = 4,
        enc_layers: int = 2,
        dec_layers: int = 2,
        mlp_ratio: float = 4.0,
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
        ndn = int(num_dn_queries)
        if ndn < 0:
            raise ValueError("num_dn_queries must be >= 0")
        dm = int(d_model)
        nh = int(num_heads)
        if dm <= 0 or nh <= 0 or dm % nh != 0:
            raise ValueError("d_model must be > 0 and divisible by num_heads")
        el = int(enc_layers)
        dl = int(dec_layers)
        if el <= 0 or dl <= 0:
            raise ValueError("enc_layers/dec_layers must be > 0")

        self.backbone = BackboneLowDet(
            in_channels=int(in_channels),
            stem_channels=int(stem_channels),
            low_channels=int(low_channels),
            det_channels=int(det_channels),
            depth=int(backbone_depth),
            act="relu",
        )

        self.semantic = nn.Sequential(
            nn.Conv2d(int(low_channels), dm, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dm, nt + ns, kernel_size=1, bias=True),
        )

        self.low_proj = nn.Conv2d(int(low_channels), dm, kernel_size=1, bias=True)
        self.det_proj = nn.Conv2d(int(det_channels), dm, kernel_size=1, bias=True)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=dm,
            nhead=nh,
            dim_feedforward=int(round(dm * float(mlp_ratio))),
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            enc_layer, num_layers=el, enable_nested_tensor=False
        )

        self.query = nn.Parameter(torch.randn(nq, dm) * 0.02)
        self.dn_query = nn.Parameter(torch.randn(max(1, ndn), dm) * 0.02) if ndn > 0 else None

        self.cross = nn.ModuleList(
            [nn.MultiheadAttention(dm, nh, batch_first=True) for _ in range(dl)]
        )
        self.norm1 = nn.ModuleList([nn.LayerNorm(dm) for _ in range(dl)])
        self.ffn = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(dm, int(round(dm * float(mlp_ratio)))),
                    nn.GELU(),
                    nn.Linear(int(round(dm * float(mlp_ratio))), dm),
                )
                for _ in range(dl)
            ]
        )
        self.norm2 = nn.ModuleList([nn.LayerNorm(dm) for _ in range(dl)])

        self.cls = nn.Linear(dm, nt)
        self.box = nn.Linear(dm, 4)
        self.mask_embed = nn.Linear(dm, dm)

        self.num_dn_queries = ndn
        self.num_thing_classes = nt
        self.num_stuff_classes = ns

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = check_nchw(x)
        b, _, h, w = x.shape
        low, det = self.backbone(x)

        semantic_logits = self.semantic(low)
        semantic_logits = F.interpolate(semantic_logits, size=(h, w), mode="nearest")

        pix = self.low_proj(low)
        tok = self.det_proj(det).flatten(2).transpose(1, 2)
        tok = self.encoder(tok)

        q_main = self.query.unsqueeze(0).expand(b, -1, -1)
        if self.num_dn_queries > 0 and self.dn_query is not None:
            q_dn = self.dn_query[: self.num_dn_queries].unsqueeze(0).expand(b, -1, -1)
            q = torch.cat([q_dn, q_main], dim=1)
        else:
            q = q_main

        for attn, n1, ffn, n2 in zip(self.cross, self.norm1, self.ffn, self.norm2, strict=True):
            hq, _ = attn(q, tok, tok, need_weights=False)
            q = n1(q + hq)
            q = n2(q + ffn(q))

        all_cls_logits = self.cls(q)
        all_boxes = torch.sigmoid(self.box(q))
        me = self.mask_embed(q)
        b, dm, h4, w4 = pix.shape
        pix_flat = pix.flatten(2)
        all_masks = torch.bmm(me, pix_flat).view(b, me.shape[1], h4, w4)
        all_masks = F.interpolate(all_masks, size=(h, w), mode="nearest")

        ndn = int(self.num_dn_queries)
        if ndn > 0:
            dn_query_cls_logits = all_cls_logits[:, :ndn]
            dn_query_boxes = all_boxes[:, :ndn]
            dn_mask_logits = all_masks[:, :ndn]
            query_cls_logits = all_cls_logits[:, ndn:]
            query_boxes = all_boxes[:, ndn:]
            mask_logits = all_masks[:, ndn:]
        else:
            dn_query_cls_logits = torch.zeros(
                b, 0, int(self.num_thing_classes), device=x.device, dtype=x.dtype
            )
            dn_query_boxes = torch.zeros(b, 0, 4, device=x.device, dtype=x.dtype)
            dn_mask_logits = torch.zeros(b, 0, h, w, device=x.device, dtype=x.dtype)
            query_cls_logits = all_cls_logits
            query_boxes = all_boxes
            mask_logits = all_masks

        scores = query_cls_logits.softmax(dim=-1).max(dim=-1).values
        panoptic_map = fuse_panoptic(
            semantic_logits, mask_logits, scores, thing_offset=int(self.num_stuff_classes)
        )

        return {
            "semantic_logits": semantic_logits,
            "query_cls_logits": query_cls_logits,
            "query_boxes": query_boxes,
            "mask_logits": mask_logits,
            "dn_query_cls_logits": dn_query_cls_logits,
            "dn_query_boxes": dn_query_boxes,
            "dn_mask_logits": dn_mask_logits,
            "panoptic_map": panoptic_map,
        }


_VARIANTS: dict[str, dict] = {
    "dn_detr_panoptic_tiny": {
        "stem": 24,
        "low": 40,
        "det": 80,
        "depth": 1,
        "queries": 16,
        "dn": 8,
        "d_model": 80,
        "heads": 4,
        "enc": 1,
        "dec": 1,
    },
    "dn_detr_panoptic_small": {
        "stem": 24,
        "low": 48,
        "det": 96,
        "depth": 2,
        "queries": 32,
        "dn": 16,
        "d_model": 96,
        "heads": 4,
        "enc": 2,
        "dec": 2,
    },
    "dn_detr_panoptic_base": {
        "stem": 32,
        "low": 64,
        "det": 128,
        "depth": 3,
        "queries": 64,
        "dn": 32,
        "d_model": 128,
        "heads": 8,
        "enc": 3,
        "dec": 3,
    },
}


def build_dn_detr_panoptic_segmenter(
    *,
    in_channels: int,
    num_thing_classes: int,
    num_stuff_classes: int,
    variant: str = "dn_detr_panoptic_small",
    width_mult: float = 1.0,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(
            f"Unknown DN-DETR-panoptic variant: {variant!r}. Supported: {sorted(_VARIANTS)}"
        )
    spec = _VARIANTS[name]

    stem = scale_channels(int(spec["stem"]), float(width_mult), min_ch=16, divisor=8)
    low = scale_channels(int(spec["low"]), float(width_mult), min_ch=16, divisor=8)
    det = scale_channels(int(spec["det"]), float(width_mult), min_ch=16, divisor=8)
    dm = scale_channels(int(spec["d_model"]), float(width_mult), min_ch=16, divisor=8)

    heads = int(spec["heads"])
    while heads > 1 and dm % heads != 0:
        heads -= 1
    queries = max(4, int(round(int(spec["queries"]) * float(width_mult))))
    dnq = max(0, int(round(int(spec["dn"]) * float(width_mult))))

    return DNDTRPanoptic(
        in_channels=int(in_channels),
        num_thing_classes=int(num_thing_classes),
        num_stuff_classes=int(num_stuff_classes),
        num_queries=int(queries),
        num_dn_queries=int(dnq),
        stem_channels=int(stem),
        low_channels=int(low),
        det_channels=int(det),
        backbone_depth=int(spec["depth"]),
        d_model=int(dm),
        num_heads=int(heads),
        enc_layers=int(spec["enc"]),
        dec_layers=int(spec["dec"]),
        mlp_ratio=4.0,
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_dn_detr_panoptic_segmenter(
        in_channels=3,
        num_thing_classes=3,
        num_stuff_classes=2,
        variant="dn_detr_panoptic_tiny",
        width_mult=0.5,
    )
    out = m(x)
    print("dn_detr_panoptic_tiny", {k: tuple(v.shape) for k, v in out.items()})
    loss = (
        out["semantic_logits"].mean()
        + out["query_cls_logits"].mean()
        + out["mask_logits"].mean()
        + out["dn_mask_logits"].mean()
    )
    loss.backward()
    print("ok")
