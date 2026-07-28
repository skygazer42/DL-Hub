import torch
from torch import nn

from dlhub.vision.backbones._blocks import scale_channels
from dlhub.vision.detection._common import BackboneC3C5, check_nchw
from dlhub.vision.detection._detr_utils import MLP, SimpleTransformer, flatten_hw


class DeformableDETRDetector(nn.Module):
    """Deformable DETR *style* (compact).

    Real Deformable DETR uses sparse sampling attention; here we approximate with
    multi-scale memory concatenation while keeping pure-torch and compact-first.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        stem_channels: int = 32,
        backbone_channels: tuple[int, int, int] = (64, 96, 128),
        backbone_depth: int = 2,
        d_model: int = 128,
        num_heads: int = 4,
        num_queries: int = 50,
        enc_layers: int = 2,
        dec_layers: int = 2,
    ) -> None:
        super().__init__()
        nc = int(num_classes)
        if nc <= 0:
            raise ValueError("num_classes must be > 0")
        dm = int(d_model)
        q = int(num_queries)
        if q <= 0:
            raise ValueError("num_queries must be > 0")

        c3, c4, c5 = (int(x) for x in backbone_channels)
        self.backbone = BackboneC3C5(
            in_channels=int(in_channels),
            stem_channels=int(stem_channels),
            c3_channels=c3,
            c4_channels=c4,
            c5_channels=c5,
            depth=int(backbone_depth),
            act="relu",
        )
        self.p3 = nn.Conv2d(c3, dm, kernel_size=1)
        self.p4 = nn.Conv2d(c4, dm, kernel_size=1)
        self.p5 = nn.Conv2d(c5, dm, kernel_size=1)

        self.transformer = SimpleTransformer(
            dim=dm,
            num_heads=int(num_heads),
            num_encoder_layers=int(enc_layers),
            num_decoder_layers=int(dec_layers),
            mlp_ratio=4.0,
            dropout=0.0,
        )
        self.query_embed = nn.Parameter(torch.randn(q, dm) * 0.02)
        self.class_head = nn.Linear(dm, nc)
        self.box_head = MLP(dm, dm, 4, num_layers=3, act="relu")

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = check_nchw(x)
        b = x.shape[0]
        c3, c4, c5 = self.backbone(x)
        m3 = flatten_hw(self.p3(c3))
        m4 = flatten_hw(self.p4(c4))
        m5 = flatten_hw(self.p5(c5))
        memory = torch.cat([m3, m4, m5], dim=1)
        queries = self.query_embed.unsqueeze(0).expand(b, -1, -1).contiguous()
        hs = self.transformer(memory, queries)
        return {"class_logits": self.class_head(hs), "boxes": torch.sigmoid(self.box_head(hs))}


_VARIANTS: dict[str, dict] = {
    "deformable_detr_tiny": {
        "stem": 24,
        "c3": 48,
        "c4": 64,
        "c5": 80,
        "depth": 1,
        "d_model": 96,
        "heads": 4,
        "q": 50,
        "enc": 1,
        "dec": 1,
    },
    "deformable_detr_small": {
        "stem": 32,
        "c3": 64,
        "c4": 96,
        "c5": 128,
        "depth": 2,
        "d_model": 128,
        "heads": 4,
        "q": 100,
        "enc": 2,
        "dec": 2,
    },
    "deformable_detr_base": {
        "stem": 48,
        "c3": 96,
        "c4": 144,
        "c5": 192,
        "depth": 3,
        "d_model": 192,
        "heads": 6,
        "q": 300,
        "enc": 3,
        "dec": 3,
    },
}


def build_deformable_detr_detector(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "deformable_detr_tiny",
    width_mult: float = 1.0,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(
            f"Unknown Deformable DETR variant: {variant!r}. Supported: {sorted(_VARIANTS)}"
        )
    spec = _VARIANTS[name]
    stem = scale_channels(int(spec["stem"]), float(width_mult), min_ch=16, divisor=8)
    c3 = scale_channels(int(spec["c3"]), float(width_mult), min_ch=16, divisor=8)
    c4 = scale_channels(int(spec["c4"]), float(width_mult), min_ch=16, divisor=8)
    c5 = scale_channels(int(spec["c5"]), float(width_mult), min_ch=16, divisor=8)
    d_model = scale_channels(int(spec["d_model"]), float(width_mult), min_ch=32, divisor=8)
    return DeformableDETRDetector(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        stem_channels=int(stem),
        backbone_channels=(int(c3), int(c4), int(c5)),
        backbone_depth=int(spec["depth"]),
        d_model=int(d_model),
        num_heads=int(spec["heads"]),
        num_queries=int(spec["q"]),
        enc_layers=int(spec["enc"]),
        dec_layers=int(spec["dec"]),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 128, 128)
    m = build_deformable_detr_detector(
        in_channels=3, num_classes=2, variant="deformable_detr_tiny", width_mult=0.5
    )
    out = m(x)
    print("deformable_detr_tiny", {k: tuple(v.shape) for k, v in out.items()})
    loss = out["class_logits"].mean() + out["boxes"].mean()
    loss.backward()
    print("ok")
