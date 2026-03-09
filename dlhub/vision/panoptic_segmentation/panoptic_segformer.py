
import torch
from torch import nn
import torch.nn.functional as F

from dlhub.vision.backbones._blocks import ConvBNAct, scale_channels
from dlhub.vision.panoptic_segmentation._common import check_nchw, fuse_panoptic


class PanopticSegFormer(nn.Module):
    """SegFormer-style panoptic segmentation (toy-first).

    Single patch embedding + Transformer encoder to produce a /4 feature map.
    Semantic logits come from a conv head; instance masks come from query cross-attn + dot-product masks.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        num_thing_classes: int,
        num_stuff_classes: int,
        embed_dim: int = 96,
        depth: int = 3,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        num_queries: int = 32,
        decoder_layers: int = 2,
    ) -> None:
        super().__init__()
        nt = int(num_thing_classes)
        ns = int(num_stuff_classes)
        if nt <= 0:
            raise ValueError("num_thing_classes must be > 0")
        if ns <= 0:
            raise ValueError("num_stuff_classes must be > 0")
        dim = int(embed_dim)
        if dim <= 0:
            raise ValueError("embed_dim must be > 0")
        h = int(num_heads)
        if h <= 0 or dim % h != 0:
            raise ValueError("num_heads must be > 0 and divide embed_dim")
        d = int(depth)
        if d <= 0:
            raise ValueError("depth must be > 0")
        nq = int(num_queries)
        if nq <= 0:
            raise ValueError("num_queries must be > 0")
        dl = int(decoder_layers)
        if dl <= 0:
            raise ValueError("decoder_layers must be > 0")

        self.patch = nn.Sequential(
            ConvBNAct(int(in_channels), dim, kernel_size=7, stride=4, padding=3, act="relu"),
        )

        ff = int(round(dim * float(mlp_ratio)))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=h,
            dim_feedforward=ff,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=d)

        self.semantic_head = nn.Sequential(
            ConvBNAct(dim, dim, kernel_size=3, stride=1, act="relu"),
            nn.Conv2d(dim, nt + ns, kernel_size=1, bias=True),
        )

        self.query = nn.Parameter(torch.randn(nq, dim) * 0.02)
        self.cross = nn.ModuleList([nn.MultiheadAttention(dim, h, batch_first=True) for _ in range(dl)])
        self.norm1 = nn.ModuleList([nn.LayerNorm(dim) for _ in range(dl)])
        self.ffn = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(dim, int(round(dim * float(mlp_ratio)))),
                    nn.GELU(),
                    nn.Linear(int(round(dim * float(mlp_ratio))), dim),
                )
                for _ in range(dl)
            ]
        )
        self.norm2 = nn.ModuleList([nn.LayerNorm(dim) for _ in range(dl)])

        self.cls = nn.Linear(dim, nt)
        self.mask_embed = nn.Linear(dim, dim)

        self.num_thing_classes = nt
        self.num_stuff_classes = ns

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = check_nchw(x)
        b, _, h, w = x.shape

        feat = self.patch(x)  # (B,D,H/4,W/4)
        b, d, h4, w4 = feat.shape
        tok = feat.permute(0, 2, 3, 1).reshape(b, h4 * w4, d)
        tok = self.encoder(tok)
        feat = tok.view(b, h4, w4, d).permute(0, 3, 1, 2).contiguous()

        semantic_logits = self.semantic_head(feat)
        semantic_logits = F.interpolate(semantic_logits, size=(h, w), mode="nearest")

        q = self.query.unsqueeze(0).expand(b, -1, -1)
        for attn, n1, ffn, n2 in zip(self.cross, self.norm1, self.ffn, self.norm2, strict=True):
            hq, _ = attn(q, tok, tok, need_weights=False)
            q = n1(q + hq)
            q = n2(q + ffn(q))

        query_cls_logits = self.cls(q)
        me = self.mask_embed(q)
        feat_flat = feat.flatten(2)  # (B,D,HW)
        mask_flat = torch.bmm(me, feat_flat)
        mask_logits = mask_flat.view(b, me.shape[1], h4, w4)
        mask_logits = F.interpolate(mask_logits, size=(h, w), mode="nearest")

        scores = query_cls_logits.softmax(dim=-1).max(dim=-1).values
        panoptic_map = fuse_panoptic(semantic_logits, mask_logits, scores, thing_offset=int(self.num_stuff_classes))

        return {
            "semantic_logits": semantic_logits,
            "query_cls_logits": query_cls_logits,
            "mask_logits": mask_logits,
            "panoptic_map": panoptic_map,
        }


_VARIANTS: dict[str, dict] = {
    "panoptic_segformer_tiny": {"embed": 64, "depth": 2, "heads": 4, "queries": 16, "dec": 1},
    "panoptic_segformer_small": {"embed": 96, "depth": 3, "heads": 4, "queries": 32, "dec": 2},
    "panoptic_segformer_base": {"embed": 128, "depth": 4, "heads": 8, "queries": 64, "dec": 3},
}


def build_panoptic_segformer_panoptic_segmenter(
    *,
    in_channels: int,
    num_thing_classes: int,
    num_stuff_classes: int,
    variant: str = "panoptic_segformer_small",
    width_mult: float = 1.0,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown Panoptic-SegFormer variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]

    dim = scale_channels(int(spec["embed"]), float(width_mult), min_ch=32, divisor=8)
    heads = int(spec["heads"])
    while heads > 1 and dim % heads != 0:
        heads -= 1
    queries = max(4, int(round(int(spec["queries"]) * float(width_mult))))

    return PanopticSegFormer(
        in_channels=int(in_channels),
        num_thing_classes=int(num_thing_classes),
        num_stuff_classes=int(num_stuff_classes),
        embed_dim=int(dim),
        depth=int(spec["depth"]),
        num_heads=int(heads),
        mlp_ratio=4.0,
        num_queries=int(queries),
        decoder_layers=int(spec["dec"]),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_panoptic_segformer_panoptic_segmenter(
        in_channels=3, num_thing_classes=3, num_stuff_classes=2, variant="panoptic_segformer_tiny", width_mult=0.5
    )
    out = m(x)
    print("panoptic_segformer_tiny", {k: tuple(v.shape) for k, v in out.items()})
    loss = out["semantic_logits"].mean() + out["query_cls_logits"].mean() + out["mask_logits"].mean()
    loss.backward()
    print("ok")

