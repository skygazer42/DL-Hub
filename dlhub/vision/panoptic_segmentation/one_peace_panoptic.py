import torch
import torch.nn.functional as F
from torch import nn

from dlhub.vision.backbones._blocks import scale_channels
from dlhub.vision.panoptic_segmentation._common import BackboneLowDet, check_nchw, fuse_panoptic


class OnePeacePanoptic(nn.Module):
    """Mask2Former-style panoptic segmentation (compact-first).

    Multi-scale pixel embedding (low + upsampled det) + query-based masks, plus a semantic head.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        num_thing_classes: int,
        num_stuff_classes: int,
        num_queries: int = 32,
        stem_channels: int = 24,
        low_channels: int = 48,
        det_channels: int = 96,
        backbone_depth: int = 2,
        d_model: int = 96,
        num_heads: int = 4,
        decoder_layers: int = 3,
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

        self.semantic = nn.Sequential(
            nn.Conv2d(int(low_channels), dm, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dm, nt + ns, kernel_size=1, bias=True),
        )

        self.low_proj = nn.Conv2d(int(low_channels), dm, kernel_size=1, bias=True)
        self.det_proj = nn.Conv2d(int(det_channels), dm, kernel_size=1, bias=True)
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

        self.cls = nn.Linear(dm, nt)
        self.mask_embed = nn.Linear(dm, dm)

        self.num_thing_classes = nt
        self.num_stuff_classes = ns

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = check_nchw(x)
        b, _, h, w = x.shape
        low, det = self.backbone(x)

        semantic_logits = self.semantic(low)
        semantic_logits = F.interpolate(semantic_logits, size=(h, w), mode="nearest")

        low_e = self.low_proj(low)
        det_e = self.det_proj(det)
        det_e_up = F.interpolate(det_e, size=low_e.shape[-2:], mode="nearest")
        pix = low_e + det_e_up  # (B,DM,H/4,W/4)

        tok = det_e.flatten(2).transpose(1, 2)  # (B,N,DM)
        q = self.query.unsqueeze(0).expand(b, -1, -1)
        for attn, n1, ffn, n2 in zip(self.cross, self.norm1, self.ffn, self.norm2, strict=True):
            hq, _ = attn(q, tok, tok, need_weights=False)
            q = n1(q + hq)
            q = n2(q + ffn(q))

        query_cls_logits = self.cls(q)
        me = self.mask_embed(q)
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
            "mask_logits": mask_logits,
            "panoptic_map": panoptic_map,
        }


_VARIANTS: dict[str, dict] = {
    "one_peace_panoptic_tiny": {
        "stem": 24,
        "low": 40,
        "det": 80,
        "depth": 1,
        "queries": 16,
        "d_model": 80,
        "heads": 4,
        "layers": 2,
    },
    "one_peace_panoptic_small": {
        "stem": 24,
        "low": 48,
        "det": 96,
        "depth": 2,
        "queries": 32,
        "d_model": 96,
        "heads": 4,
        "layers": 3,
    },
    "one_peace_panoptic_base": {
        "stem": 32,
        "low": 64,
        "det": 128,
        "depth": 3,
        "queries": 64,
        "d_model": 128,
        "heads": 8,
        "layers": 4,
    },
}


def build_one_peace_panoptic_segmenter(
    *,
    in_channels: int,
    num_thing_classes: int,
    num_stuff_classes: int,
    variant: str = "one_peace_panoptic_small",
    width_mult: float = 1.0,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(
            f"Unknown Mask2Former-panoptic variant: {variant!r}. Supported: {sorted(_VARIANTS)}"
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

    return OnePeacePanoptic(
        in_channels=int(in_channels),
        num_thing_classes=int(num_thing_classes),
        num_stuff_classes=int(num_stuff_classes),
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
    m = build_one_peace_panoptic_segmenter(
        in_channels=3,
        num_thing_classes=3,
        num_stuff_classes=2,
        variant="one_peace_panoptic_tiny",
        width_mult=0.5,
    )
    out = m(x)
    print("one_peace_panoptic_tiny", {k: tuple(v.shape) for k, v in out.items()})
    loss = (
        out["semantic_logits"].mean() + out["query_cls_logits"].mean() + out["mask_logits"].mean()
    )
    loss.backward()
    print("ok")
