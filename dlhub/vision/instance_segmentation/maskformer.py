import torch
from torch import nn

from dlhub.vision.backbones._blocks import scale_channels
from dlhub.vision.instance_segmentation._common import BackboneLowDet, check_nchw


class MaskFormer(nn.Module):
    """MaskFormer-style instance segmentation (toy-first).

    Query-based class prediction + mask prediction via dot-product with a pixel embedding map.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        num_queries: int = 32,
        stem_channels: int = 24,
        low_channels: int = 48,
        det_channels: int = 96,
        backbone_depth: int = 2,
        d_model: int = 96,
        num_heads: int = 4,
        decoder_layers: int = 2,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        nc = int(num_classes)
        if nc <= 0:
            raise ValueError("num_classes must be > 0")
        nq = int(num_queries)
        if nq <= 0:
            raise ValueError("num_queries must be > 0")
        dm = int(d_model)
        nh = int(num_heads)
        if dm <= 0 or nh <= 0 or dm % nh != 0:
            raise ValueError("d_model must be > 0 and divisible by num_heads")
        dl = int(decoder_layers)
        if dl <= 0:
            raise ValueError("decoder_layers must be > 0")

        self.backbone = BackboneLowDet(
            in_channels=int(in_channels),
            stem_channels=int(stem_channels),
            low_channels=int(low_channels),
            det_channels=int(det_channels),
            depth=int(backbone_depth),
            act="relu",
        )

        self.pixel_proj = nn.Conv2d(int(low_channels), dm, kernel_size=1, bias=True)
        self.token_proj = nn.Conv2d(int(det_channels), dm, kernel_size=1, bias=True)
        self.query = nn.Parameter(torch.randn(nq, dm) * 0.02)

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

        self.cls = nn.Linear(dm, nc)
        self.mask_embed = nn.Linear(dm, dm)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = check_nchw(x)
        low, det = self.backbone(x)

        pix = self.pixel_proj(low)  # (B,DM,H/4,W/4)
        tok = self.token_proj(det).flatten(2).transpose(1, 2)  # (B,N,DM)

        b = tok.shape[0]
        q = self.query.unsqueeze(0).expand(b, -1, -1)  # (B,Q,DM)
        for attn, n1, ffn, n2 in zip(self.cross, self.norm1, self.ffn, self.norm2, strict=True):
            h, _ = attn(q, tok, tok, need_weights=False)
            q = n1(q + h)
            q = n2(q + ffn(q))

        cls_logits = self.cls(q)  # (B,Q,C)
        me = self.mask_embed(q)  # (B,Q,DM)
        b, dm, h4, w4 = pix.shape
        pix_flat = pix.flatten(2)  # (B,DM,HW)
        mask_flat = torch.bmm(me, pix_flat)  # (B,Q,HW)
        mask_logits = mask_flat.view(b, me.shape[1], h4, w4)
        return {"query_cls_logits": cls_logits, "mask_logits": mask_logits}


_VARIANTS: dict[str, dict] = {
    "maskformer_tiny": {
        "stem": 24,
        "low": 40,
        "det": 80,
        "depth": 1,
        "queries": 16,
        "d_model": 80,
        "heads": 4,
        "layers": 1,
    },
    "maskformer_small": {
        "stem": 24,
        "low": 48,
        "det": 96,
        "depth": 2,
        "queries": 32,
        "d_model": 96,
        "heads": 4,
        "layers": 2,
    },
    "maskformer_base": {
        "stem": 32,
        "low": 64,
        "det": 128,
        "depth": 3,
        "queries": 64,
        "d_model": 128,
        "heads": 8,
        "layers": 3,
    },
}


def build_maskformer_instance_segmenter(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "maskformer_small",
    width_mult: float = 1.0,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown MaskFormer variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]

    stem = scale_channels(int(spec["stem"]), float(width_mult), min_ch=16, divisor=8)
    low = scale_channels(int(spec["low"]), float(width_mult), min_ch=16, divisor=8)
    det = scale_channels(int(spec["det"]), float(width_mult), min_ch=16, divisor=8)
    dm = scale_channels(int(spec["d_model"]), float(width_mult), min_ch=16, divisor=8)
    heads = int(spec["heads"])
    while heads > 1 and dm % heads != 0:
        heads -= 1
    queries = max(4, int(round(int(spec["queries"]) * float(width_mult))))

    return MaskFormer(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        num_queries=int(queries),
        stem_channels=int(stem),
        low_channels=int(low),
        det_channels=int(det),
        backbone_depth=int(spec["depth"]),
        d_model=int(dm),
        num_heads=int(heads),
        decoder_layers=int(spec["layers"]),
        mlp_ratio=4.0,
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_maskformer_instance_segmenter(
        in_channels=3, num_classes=3, variant="maskformer_tiny", width_mult=0.5
    )
    out = m(x)
    print("maskformer_tiny", {k: tuple(v.shape) for k, v in out.items()})
    loss = sum(v.mean() for v in out.values())
    loss.backward()
    print("ok")
