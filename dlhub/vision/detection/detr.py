import torch
from torch import nn

from dlhub.vision.backbones._blocks import ConvBNAct, scale_channels
from dlhub.vision.detection._detr_utils import (
    MLP,
    SimpleTransformer,
    flatten_hw,
    sine_positional_encoding_1d,
)


class _ConvBackboneStride8(nn.Module):
    """Tiny conv backbone producing a stride-8 feature map for DETR-like models."""

    def __init__(
        self, *, in_channels: int, stem_channels: int, feat_channels: int, depth: int
    ) -> None:
        super().__init__()
        c_in = int(in_channels)
        stem = int(stem_channels)
        feat = int(feat_channels)
        d = int(depth)
        if d <= 0:
            raise ValueError("depth must be > 0")
        layers: list[nn.Module] = [
            ConvBNAct(c_in, stem, kernel_size=3, stride=2, act="relu"),  # /2
            ConvBNAct(stem, stem, kernel_size=3, stride=2, act="relu"),  # /4
            ConvBNAct(stem, feat, kernel_size=3, stride=2, act="relu"),  # /8
        ]
        for _ in range(d):
            layers.append(ConvBNAct(feat, feat, kernel_size=3, stride=1, act="relu"))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DETRDetector(nn.Module):
    """DETR-style query-based detector (toy-first).

    Output:
      - class_logits: (B, Q, num_classes)
      - boxes: (B, Q, 4) in [0,1] via sigmoid (cx, cy, w, h)
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
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        nc = int(num_classes)
        if nc <= 0:
            raise ValueError("num_classes must be > 0")
        dm = int(d_model)
        if dm <= 0:
            raise ValueError("d_model must be > 0")
        q = int(num_queries)
        if q <= 0:
            raise ValueError("num_queries must be > 0")
        if dm % 2 != 0:
            raise ValueError("d_model must be even for sinusoidal encoding")

        self.backbone = _ConvBackboneStride8(
            in_channels=int(in_channels),
            stem_channels=int(stem_channels),
            feat_channels=int(feat_channels),
            depth=int(backbone_depth),
        )
        self.proj = nn.Conv2d(int(feat_channels), dm, kernel_size=1)

        self.transformer = SimpleTransformer(
            dim=dm,
            num_heads=int(num_heads),
            num_encoder_layers=int(enc_layers),
            num_decoder_layers=int(dec_layers),
            mlp_ratio=float(mlp_ratio),
            dropout=float(dropout),
        )
        self.query_embed = nn.Parameter(torch.randn(q, dm) * 0.02)
        self.class_head = nn.Linear(dm, nc)
        self.box_head = MLP(dm, dm, 4, num_layers=3, act="relu")

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = x.to(torch.float32)
        if x.ndim != 4:
            raise ValueError(f"Expected input shape (B, C, H, W), got {tuple(x.shape)}")
        b = x.shape[0]
        feat = self.proj(self.backbone(x))  # (B, D, H', W')
        memory = flatten_hw(feat)  # (B, N, D)

        # A simple 1D positional encoding over flattened tokens (toy).
        pos = sine_positional_encoding_1d(memory.shape[1], memory.shape[2], device=memory.device)
        memory = memory + pos.unsqueeze(0)

        queries = self.query_embed.unsqueeze(0).expand(b, -1, -1).contiguous()
        hs = self.transformer(memory, queries)  # (B, Q, D)
        class_logits = self.class_head(hs)
        boxes = torch.sigmoid(self.box_head(hs))
        return {"class_logits": class_logits, "boxes": boxes}


_VARIANTS: dict[str, dict] = {
    "detr_tiny": {
        "stem": 24,
        "feat": 96,
        "depth": 1,
        "d_model": 96,
        "heads": 4,
        "q": 32,
        "enc": 1,
        "dec": 1,
    },
    "detr_small": {
        "stem": 32,
        "feat": 128,
        "depth": 2,
        "d_model": 128,
        "heads": 4,
        "q": 50,
        "enc": 2,
        "dec": 2,
    },
    "detr_base": {
        "stem": 48,
        "feat": 192,
        "depth": 2,
        "d_model": 192,
        "heads": 6,
        "q": 100,
        "enc": 3,
        "dec": 3,
    },
}


def build_detr_detector(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "detr_tiny",
    width_mult: float = 1.0,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown DETR variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]

    stem = scale_channels(int(spec["stem"]), float(width_mult), min_ch=16, divisor=8)
    feat = scale_channels(int(spec["feat"]), float(width_mult), min_ch=16, divisor=8)
    d_model = scale_channels(int(spec["d_model"]), float(width_mult), min_ch=32, divisor=8)
    if int(d_model) % 2 != 0:
        d_model += 8  # keep even

    return DETRDetector(
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
    m = build_detr_detector(in_channels=3, num_classes=2, variant="detr_tiny", width_mult=0.5)
    out = m(x)
    print("detr_tiny", {k: tuple(v.shape) for k, v in out.items()})
    loss = out["class_logits"].mean() + out["boxes"].mean()
    loss.backward()
    print("ok")
