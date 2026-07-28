import torch
from torch import nn

from dlhub.vision.backbones._blocks import scale_channels
from dlhub.vision.detection._common import BackboneC3C5, check_nchw
from dlhub.vision.detection._detr_utils import MLP, SimpleTransformer, flatten_hw


class DINODetector(nn.Module):
    """DINO-style query detector (compact).

    This keeps two key DINO ideas in a lightweight form:
    - anchor-like reference boxes refined by decoder outputs
    - an explicit denoising query branch returned alongside the main predictions
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
        num_dn_queries: int = 25,
        enc_layers: int = 2,
        dec_layers: int = 2,
    ) -> None:
        super().__init__()
        nc = int(num_classes)
        if nc <= 0:
            raise ValueError("num_classes must be > 0")
        dm = int(d_model)
        if dm <= 0:
            raise ValueError("d_model must be > 0")
        q = int(num_queries)
        dn_q = int(num_dn_queries)
        if q <= 0 or dn_q <= 0:
            raise ValueError("num_queries and num_dn_queries must be > 0")

        c3, c4, c5 = (int(v) for v in backbone_channels)
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
        self.dn_query_embed = nn.Parameter(torch.randn(dn_q, dm) * 0.02)
        self.ref_boxes = nn.Parameter(torch.rand(q, 4) * 0.5)
        self.dn_ref_boxes = nn.Parameter(torch.rand(dn_q, 4) * 0.5)
        self.class_head = nn.Linear(dm, nc)
        self.box_head = MLP(dm, dm, 4, num_layers=3, act="relu")

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = check_nchw(x)
        b = x.shape[0]
        c3, c4, c5 = self.backbone(x)
        memory = torch.cat(
            [flatten_hw(self.p3(c3)), flatten_hw(self.p4(c4)), flatten_hw(self.p5(c5))],
            dim=1,
        )

        global_token = memory.mean(dim=1, keepdim=True)
        dn_queries = self.dn_query_embed.unsqueeze(0).expand(b, -1, -1).contiguous()
        dn_queries = dn_queries + 0.1 * global_token
        main_queries = self.query_embed.unsqueeze(0).expand(b, -1, -1).contiguous()
        hs = self.transformer(memory, torch.cat([dn_queries, main_queries], dim=1))

        dn_count = self.dn_query_embed.shape[0]
        dn_hs = hs[:, :dn_count]
        main_hs = hs[:, dn_count:]

        dn_class_logits = self.class_head(dn_hs)
        dn_boxes = torch.sigmoid(self.dn_ref_boxes.unsqueeze(0) + self.box_head(dn_hs))
        class_logits = self.class_head(main_hs)
        boxes = torch.sigmoid(self.ref_boxes.unsqueeze(0) + self.box_head(main_hs))
        return {
            "class_logits": class_logits,
            "boxes": boxes,
            "dn_class_logits": dn_class_logits,
            "dn_boxes": dn_boxes,
        }


_VARIANTS: dict[str, dict[str, int]] = {
    "dino_tiny": {
        "stem": 24,
        "c3": 48,
        "c4": 64,
        "c5": 80,
        "depth": 1,
        "d_model": 96,
        "heads": 4,
        "q": 50,
        "dn_q": 25,
        "enc": 1,
        "dec": 1,
    },
    "dino_small": {
        "stem": 32,
        "c3": 64,
        "c4": 96,
        "c5": 128,
        "depth": 2,
        "d_model": 128,
        "heads": 4,
        "q": 100,
        "dn_q": 50,
        "enc": 2,
        "dec": 2,
    },
    "dino_base": {
        "stem": 48,
        "c3": 96,
        "c4": 144,
        "c5": 192,
        "depth": 3,
        "d_model": 192,
        "heads": 6,
        "q": 300,
        "dn_q": 100,
        "enc": 3,
        "dec": 3,
    },
}


def build_dino_detector(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "dino_tiny",
    width_mult: float = 1.0,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown DINO variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]

    stem = scale_channels(int(spec["stem"]), float(width_mult), min_ch=16, divisor=8)
    c3 = scale_channels(int(spec["c3"]), float(width_mult), min_ch=16, divisor=8)
    c4 = scale_channels(int(spec["c4"]), float(width_mult), min_ch=16, divisor=8)
    c5 = scale_channels(int(spec["c5"]), float(width_mult), min_ch=16, divisor=8)
    d_model = scale_channels(int(spec["d_model"]), float(width_mult), min_ch=32, divisor=8)
    heads = int(spec["heads"])
    if d_model % heads != 0:
        d_model = max(heads, heads * round(d_model / heads))

    return DINODetector(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        stem_channels=int(stem),
        backbone_channels=(int(c3), int(c4), int(c5)),
        backbone_depth=int(spec["depth"]),
        d_model=int(d_model),
        num_heads=heads,
        num_queries=int(spec["q"]),
        num_dn_queries=int(spec["dn_q"]),
        enc_layers=int(spec["enc"]),
        dec_layers=int(spec["dec"]),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 128, 128)
    m = build_dino_detector(in_channels=3, num_classes=2, variant="dino_tiny", width_mult=0.5)
    out = m(x)
    print("dino_tiny", {k: tuple(v.shape) for k, v in out.items()})
    loss = sum(v.mean() for v in out.values())
    loss.backward()
    print("ok")
