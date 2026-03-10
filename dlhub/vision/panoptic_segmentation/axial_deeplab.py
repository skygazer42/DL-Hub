import torch
import torch.nn.functional as F
from torch import nn

from dlhub.vision.backbones._blocks import ConvBNAct, scale_channels
from dlhub.vision.panoptic_segmentation._common import (
    BackboneC2C3C4C5,
    ProtoNet,
    check_nchw,
    masks_from_prototypes,
)


class _AxialBlock(nn.Module):
    def __init__(self, dim: int, *, num_heads: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        d = int(dim)
        h = int(num_heads)
        if d <= 0:
            raise ValueError("dim must be > 0")
        if h <= 0 or d % h != 0:
            raise ValueError("num_heads must be > 0 and divide dim")

        self.attn_h = nn.MultiheadAttention(d, h, batch_first=True)
        self.attn_w = nn.MultiheadAttention(d, h, batch_first=True)
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(
            nn.Linear(d, int(round(d * float(mlp_ratio)))),
            nn.GELU(),
            nn.Linear(int(round(d * float(mlp_ratio))), d),
        )
        self.norm3 = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        b, c, h, w = x.shape
        # Height-axis attention (per column).
        t = x.permute(0, 3, 2, 1).reshape(b * w, h, c)  # (B*W, H, C)
        a, _ = self.attn_h(t, t, t, need_weights=False)
        t = self.norm1(t + a)
        t = self.norm2(t + self.mlp(t))
        xh = t.reshape(b, w, h, c).permute(0, 3, 2, 1).contiguous()

        # Width-axis attention (per row).
        t2 = xh.permute(0, 2, 3, 1).reshape(b * h, w, c)  # (B*H, W, C)
        a2, _ = self.attn_w(t2, t2, t2, need_weights=False)
        t2 = self.norm3(t2 + a2)
        xw = t2.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        return xw


class AxialDeepLabPanoptic(nn.Module):
    """Axial-DeepLab style panoptic segmentation (toy-first).

    Uses axial attention blocks on a low-resolution feature map and predicts:
    - semantic logits
    - instance masks via prototypes (toy convenience)
    """

    def __init__(
        self,
        *,
        in_channels: int,
        num_thing_classes: int,
        num_stuff_classes: int,
        stem_channels: int = 24,
        c2_channels: int = 32,
        c3_channels: int = 64,
        c4_channels: int = 96,
        c5_channels: int = 128,
        depth: int = 2,
        embed_dim: int = 96,
        axial_depth: int = 2,
        num_heads: int = 4,
        num_instances: int = 32,
        num_protos: int = 32,
    ) -> None:
        super().__init__()
        nt = int(num_thing_classes)
        ns = int(num_stuff_classes)
        if nt <= 0 or ns <= 0:
            raise ValueError("num_thing_classes/num_stuff_classes must be > 0")
        dm = int(embed_dim)
        if dm <= 0:
            raise ValueError("embed_dim must be > 0")
        ad = int(axial_depth)
        if ad <= 0:
            raise ValueError("axial_depth must be > 0")
        ni = int(num_instances)
        np = int(num_protos)
        if ni <= 0 or np <= 0:
            raise ValueError("num_instances/num_protos must be > 0")

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

        self.proj = ConvBNAct(int(c4_channels), dm, kernel_size=1, stride=1, padding=0, act="relu")
        self.axial = nn.Sequential(*[_AxialBlock(dm, num_heads=int(num_heads)) for _ in range(ad)])
        self.semantic = nn.Sequential(
            ConvBNAct(dm, dm, kernel_size=3, stride=1, act="relu"),
            nn.Conv2d(dm, nt + ns, kernel_size=1, bias=True),
        )

        # Toy instance masks from /8 features.
        self.inst_proj = ConvBNAct(
            int(c3_channels), dm, kernel_size=1, stride=1, padding=0, act="relu"
        )
        self.proto = ProtoNet(dm, np, depth=3, act="relu")
        self.query = nn.Parameter(torch.randn(ni, dm) * 0.02)
        self.q_proj = nn.Linear(dm, dm)
        self.q_cls = nn.Linear(dm, nt)
        self.q_coeff = nn.Linear(dm, np)

        self.num_instances = ni

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = check_nchw(x)
        b, _, h, w = x.shape
        _, c3, c4, _ = self.backbone(x)

        feat = self.proj(c4)
        feat = self.axial(feat)
        semantic_logits = self.semantic(feat)
        semantic_logits = F.interpolate(semantic_logits, size=(h, w), mode="nearest")

        inst_feat = self.inst_proj(c3)
        proto = self.proto(inst_feat)
        pooled = F.adaptive_avg_pool2d(inst_feat, (1, 1)).view(b, -1)
        base = torch.relu(self.q_proj(pooled)).unsqueeze(1)
        q = self.query.unsqueeze(0).expand(b, -1, -1)
        hq = base + q
        query_cls_logits = self.q_cls(hq)
        coeff = self.q_coeff(hq)
        mask_logits = masks_from_prototypes(proto, coeff)
        mask_logits = F.interpolate(mask_logits, size=(h, w), mode="nearest")

        return {
            "semantic_logits": semantic_logits,
            "query_cls_logits": query_cls_logits,
            "mask_logits": mask_logits,
        }


_VARIANTS: dict[str, dict] = {
    "axial_deeplab_tiny": {
        "stem": 24,
        "c2": 24,
        "c3": 48,
        "c4": 64,
        "c5": 96,
        "depth": 1,
        "embed": 64,
        "axial_depth": 1,
        "heads": 4,
        "instances": 16,
        "protos": 16,
    },
    "axial_deeplab_small": {
        "stem": 24,
        "c2": 32,
        "c3": 64,
        "c4": 96,
        "c5": 128,
        "depth": 2,
        "embed": 96,
        "axial_depth": 2,
        "heads": 4,
        "instances": 32,
        "protos": 32,
    },
    "axial_deeplab_base": {
        "stem": 32,
        "c2": 40,
        "c3": 80,
        "c4": 128,
        "c5": 160,
        "depth": 2,
        "embed": 128,
        "axial_depth": 3,
        "heads": 8,
        "instances": 64,
        "protos": 48,
    },
}


def build_axial_deeplab_panoptic_segmenter(
    *,
    in_channels: int,
    num_thing_classes: int,
    num_stuff_classes: int,
    variant: str = "axial_deeplab_small",
    width_mult: float = 1.0,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(
            f"Unknown Axial-DeepLab panoptic variant: {variant!r}. Supported: {sorted(_VARIANTS)}"
        )
    spec = _VARIANTS[name]

    def sc(v: int, *, min_ch: int = 16) -> int:
        return scale_channels(int(v), float(width_mult), min_ch=min_ch, divisor=8)

    stem = sc(spec["stem"])
    c2 = sc(spec["c2"])
    c3 = sc(spec["c3"])
    c4 = sc(spec["c4"])
    c5 = sc(spec["c5"])
    embed = sc(spec["embed"], min_ch=32)
    protos = max(8, int(round(int(spec["protos"]) * float(width_mult))))
    instances = max(4, int(round(int(spec["instances"]) * float(width_mult))))

    return AxialDeepLabPanoptic(
        in_channels=int(in_channels),
        num_thing_classes=int(num_thing_classes),
        num_stuff_classes=int(num_stuff_classes),
        stem_channels=int(stem),
        c2_channels=int(c2),
        c3_channels=int(c3),
        c4_channels=int(c4),
        c5_channels=int(c5),
        depth=int(spec["depth"]),
        embed_dim=int(embed),
        axial_depth=int(spec["axial_depth"]),
        num_heads=int(spec["heads"]),
        num_instances=int(instances),
        num_protos=int(protos),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_axial_deeplab_panoptic_segmenter(
        in_channels=3,
        num_thing_classes=3,
        num_stuff_classes=2,
        variant="axial_deeplab_tiny",
        width_mult=0.5,
    )
    out = m(x)
    print("axial_deeplab_tiny", {k: tuple(v.shape) for k, v in out.items()})
    loss = (
        out["semantic_logits"].mean() + out["query_cls_logits"].mean() + out["mask_logits"].mean()
    )
    loss.backward()
    print("ok")
