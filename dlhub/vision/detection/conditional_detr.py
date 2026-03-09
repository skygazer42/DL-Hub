
import torch
from torch import nn

from dlhub.vision.backbones._blocks import ConvBNAct, scale_channels
from dlhub.vision.detection._detr_utils import MLP, SimpleTransformer, flatten_hw


class ConditionalDETRDetector(nn.Module):
    """Conditional DETR *style* (toy).

    We approximate conditional queries by predicting a query positional embedding.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        stem_channels: int = 32,
        feat_channels: int = 128,
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

        self.backbone = nn.Sequential(
            ConvBNAct(int(in_channels), int(stem_channels), kernel_size=3, stride=2, act="relu"),  # /2
            ConvBNAct(int(stem_channels), int(stem_channels), kernel_size=3, stride=2, act="relu"),  # /4
            ConvBNAct(int(stem_channels), int(feat_channels), kernel_size=3, stride=2, act="relu"),  # /8
            *[ConvBNAct(int(feat_channels), int(feat_channels), kernel_size=3, stride=1, act="relu") for _ in range(int(backbone_depth))],
        )
        self.proj = nn.Conv2d(int(feat_channels), dm, kernel_size=1)

        self.transformer = SimpleTransformer(
            dim=dm,
            num_heads=int(num_heads),
            num_encoder_layers=int(enc_layers),
            num_decoder_layers=int(dec_layers),
            mlp_ratio=4.0,
            dropout=0.0,
        )
        self.query_embed = nn.Parameter(torch.randn(q, dm) * 0.02)
        self.query_pos = nn.Linear(dm, dm, bias=True)

        self.class_head = nn.Linear(dm, nc)
        self.box_head = MLP(dm, dm, 4, num_layers=3, act="relu")

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = x.to(torch.float32)
        if x.ndim != 4:
            raise ValueError(f"Expected input shape (B, C, H, W), got {tuple(x.shape)}")
        b = x.shape[0]
        feat = self.proj(self.backbone(x))
        memory = flatten_hw(feat)

        q = self.query_embed.unsqueeze(0).expand(b, -1, -1).contiguous()
        q = q + torch.tanh(self.query_pos(q))
        hs = self.transformer(memory, q)
        return {"class_logits": self.class_head(hs), "boxes": torch.sigmoid(self.box_head(hs))}


_VARIANTS: dict[str, dict] = {
    "conditional_detr_tiny": {"stem": 24, "feat": 96, "depth": 1, "d_model": 96, "heads": 4, "q": 32, "enc": 1, "dec": 1},
    "conditional_detr_small": {"stem": 32, "feat": 128, "depth": 2, "d_model": 128, "heads": 4, "q": 50, "enc": 2, "dec": 2},
    "conditional_detr_base": {"stem": 48, "feat": 192, "depth": 2, "d_model": 192, "heads": 6, "q": 100, "enc": 3, "dec": 3},
}


def build_conditional_detr_detector(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "conditional_detr_tiny",
    width_mult: float = 1.0,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown Conditional DETR variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]

    stem = scale_channels(int(spec["stem"]), float(width_mult), min_ch=16, divisor=8)
    feat = scale_channels(int(spec["feat"]), float(width_mult), min_ch=16, divisor=8)
    d_model = scale_channels(int(spec["d_model"]), float(width_mult), min_ch=32, divisor=8)

    return ConditionalDETRDetector(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        stem_channels=int(stem),
        feat_channels=int(feat),
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
    m = build_conditional_detr_detector(in_channels=3, num_classes=2, variant="conditional_detr_tiny", width_mult=0.5)
    out = m(x)
    print("conditional_detr_tiny", {k: tuple(v.shape) for k, v in out.items()})
    loss = out["class_logits"].mean() + out["boxes"].mean()
    loss.backward()
    print("ok")

