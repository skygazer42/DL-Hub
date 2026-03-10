import torch
import torch.nn.functional as F
from torch import nn

from dlhub.vision.backbones._blocks import ConvBNAct, scale_channels
from dlhub.vision.panoptic_segmentation._common import check_nchw, fuse_panoptic


class SETRPanoptic(nn.Module):
    """SETR-style panoptic segmentation (toy-first).

    Pure Transformer encoder over patch tokens + lightweight decoder to semantic logits and query masks.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        num_thing_classes: int,
        num_stuff_classes: int,
        embed_dim: int = 128,
        depth: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        patch_size: int = 16,
        num_instances: int = 32,
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
        d = int(depth)
        if d <= 0:
            raise ValueError("depth must be > 0")
        h = int(num_heads)
        if h <= 0 or dim % h != 0:
            raise ValueError("num_heads must be > 0 and divide embed_dim")
        ps = int(patch_size)
        if ps <= 0:
            raise ValueError("patch_size must be > 0")
        n = int(num_instances)
        if n <= 0:
            raise ValueError("num_instances must be > 0")

        self.patch = nn.Sequential(
            ConvBNAct(int(in_channels), dim, kernel_size=ps, stride=ps, padding=0, act="relu"),
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

        self.decoder = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            ConvBNAct(dim, dim, kernel_size=3, stride=1, act="relu"),
        )
        self.semantic_head = nn.Conv2d(dim, nt + ns, kernel_size=1, bias=True)

        self.query = nn.Parameter(torch.randn(n, dim) * 0.02)
        self.cls = nn.Linear(dim, nt)
        self.mask_embed = nn.Linear(dim, dim)

        self.patch_size = ps
        self.num_instances = n
        self.num_thing_classes = nt
        self.num_stuff_classes = ns

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = check_nchw(x)
        b, _, h, w = x.shape
        feat = self.patch(x)  # (B,D,H/ps,W/ps)
        b, d, hp, wp = feat.shape
        tok = feat.permute(0, 2, 3, 1).reshape(b, hp * wp, d)
        tok = self.encoder(tok)
        feat = tok.view(b, hp, wp, d).permute(0, 3, 1, 2).contiguous()

        # Decode to a /4 map for efficiency.
        h4, w4 = max(1, h // 4), max(1, w // 4)
        pix = self.decoder(F.interpolate(feat, size=(h4, w4), mode="nearest"))

        semantic_logits4 = self.semantic_head(pix)
        semantic_logits = F.interpolate(semantic_logits4, size=(h, w), mode="nearest")

        pooled = F.adaptive_avg_pool2d(pix, (1, 1)).flatten(1).unsqueeze(1)
        q = self.query.unsqueeze(0).expand(b, -1, -1) + pooled
        query_cls_logits = self.cls(q)
        me = self.mask_embed(q)
        mask_flat = torch.bmm(me, pix.flatten(2))
        mask_logits4 = mask_flat.view(b, int(self.num_instances), h4, w4)
        mask_logits = F.interpolate(mask_logits4, size=(h, w), mode="nearest")

        scores = query_cls_logits.softmax(dim=-1).max(dim=-1).values
        panoptic_map = fuse_panoptic(
            semantic_logits, mask_logits, scores, thing_offset=int(self.num_stuff_classes)
        )

        return {
            "semantic_logits": semantic_logits,
            "query_cls_logits": query_cls_logits,
            "mask_logits": mask_logits,
            "panoptic_map": panoptic_map,
        }


_VARIANTS: dict[str, dict] = {
    "setr_panoptic_tiny": {"embed": 64, "depth": 2, "heads": 4, "patch": 16, "instances": 16},
    "setr_panoptic_small": {"embed": 96, "depth": 3, "heads": 4, "patch": 16, "instances": 32},
    "setr_panoptic_base": {"embed": 128, "depth": 4, "heads": 8, "patch": 16, "instances": 64},
}


def build_setr_panoptic_segmenter(
    *,
    in_channels: int,
    num_thing_classes: int,
    num_stuff_classes: int,
    variant: str = "setr_panoptic_small",
    width_mult: float = 1.0,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(
            f"Unknown SETR-panoptic variant: {variant!r}. Supported: {sorted(_VARIANTS)}"
        )
    spec = _VARIANTS[name]

    dim = scale_channels(int(spec["embed"]), float(width_mult), min_ch=32, divisor=8)
    heads = int(spec["heads"])
    while heads > 1 and dim % heads != 0:
        heads -= 1
    instances = max(4, int(round(int(spec["instances"]) * float(width_mult))))

    return SETRPanoptic(
        in_channels=int(in_channels),
        num_thing_classes=int(num_thing_classes),
        num_stuff_classes=int(num_stuff_classes),
        embed_dim=int(dim),
        depth=int(spec["depth"]),
        num_heads=int(heads),
        mlp_ratio=4.0,
        patch_size=int(spec["patch"]),
        num_instances=int(instances),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_setr_panoptic_segmenter(
        in_channels=3,
        num_thing_classes=3,
        num_stuff_classes=2,
        variant="setr_panoptic_tiny",
        width_mult=0.5,
    )
    out = m(x)
    print("setr_panoptic_tiny", {k: tuple(v.shape) for k, v in out.items()})
    loss = (
        out["semantic_logits"].mean() + out["query_cls_logits"].mean() + out["mask_logits"].mean()
    )
    loss.backward()
    print("ok")
