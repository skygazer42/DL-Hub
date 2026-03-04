from __future__ import annotations

import torch
from torch import nn

from dlhub.vision.backbones._blocks import ConvBNAct, scale_channels
from dlhub.vision.detection._detr_utils import MLP, SimpleTransformer, flatten_hw


class SparseRCNNDetector(nn.Module):
    """Sparse R-CNN *style* (toy).

    Implements a small query-based ROI refinement loop without external ops.
    Output:
      - class_logits: list[(B,Q,C)] across stages
      - boxes: list[(B,Q,4)] across stages
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
        num_queries: int = 100,
        stages: int = 3,
    ) -> None:
        super().__init__()
        nc = int(num_classes)
        if nc <= 0:
            raise ValueError("num_classes must be > 0")
        dm = int(d_model)
        q = int(num_queries)
        s = int(stages)
        if q <= 0:
            raise ValueError("num_queries must be > 0")
        if s <= 0:
            raise ValueError("stages must be > 0")

        self.backbone = nn.Sequential(
            ConvBNAct(int(in_channels), int(stem_channels), kernel_size=3, stride=2, act="relu"),  # /2
            ConvBNAct(int(stem_channels), int(stem_channels), kernel_size=3, stride=2, act="relu"),  # /4
            ConvBNAct(int(stem_channels), int(feat_channels), kernel_size=3, stride=2, act="relu"),  # /8
            *[ConvBNAct(int(feat_channels), int(feat_channels), kernel_size=3, stride=1, act="relu") for _ in range(int(backbone_depth))],
        )
        self.proj = nn.Conv2d(int(feat_channels), dm, kernel_size=1)

        self.transformer = SimpleTransformer(dim=dm, num_heads=int(num_heads), num_encoder_layers=1, num_decoder_layers=1, mlp_ratio=4.0, dropout=0.0)
        self.query_embed = nn.Parameter(torch.randn(q, dm) * 0.02)
        self.proposal_boxes = nn.Parameter(torch.rand(q, 4) * 0.5)
        self.stages = s

        self.class_heads = nn.ModuleList([nn.Linear(dm, nc) for _ in range(s)])
        self.delta_heads = nn.ModuleList([MLP(dm, dm, 4, num_layers=3, act="relu") for _ in range(s)])
        self.query_update = nn.ModuleList([nn.Linear(dm, dm) for _ in range(s)])

    def forward(self, x: torch.Tensor) -> dict[str, list[torch.Tensor]]:
        x = x.to(torch.float32)
        if x.ndim != 4:
            raise ValueError(f"Expected input shape (B, C, H, W), got {tuple(x.shape)}")
        b = x.shape[0]
        feat = self.proj(self.backbone(x))
        memory = flatten_hw(feat)

        q = self.query_embed.unsqueeze(0).expand(b, -1, -1).contiguous()
        boxes = self.proposal_boxes.unsqueeze(0).expand(b, -1, -1).contiguous()

        class_out: list[torch.Tensor] = []
        box_out: list[torch.Tensor] = []
        for i in range(self.stages):
            hs = self.transformer(memory, q)
            class_logits = self.class_heads[i](hs)
            delta = self.delta_heads[i](hs)
            boxes = torch.sigmoid(boxes + delta)
            q = q + torch.tanh(self.query_update[i](hs))
            class_out.append(class_logits)
            box_out.append(boxes)
        return {"class_logits": class_out, "boxes": box_out}


_VARIANTS: dict[str, dict] = {
    "sparse_rcnn_tiny": {"stem": 24, "feat": 96, "depth": 1, "d_model": 96, "heads": 4, "q": 50, "stages": 2},
    "sparse_rcnn_small": {"stem": 32, "feat": 128, "depth": 2, "d_model": 128, "heads": 4, "q": 100, "stages": 3},
    "sparse_rcnn_base": {"stem": 48, "feat": 192, "depth": 2, "d_model": 192, "heads": 6, "q": 300, "stages": 6},
}


def build_sparse_rcnn_detector(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "sparse_rcnn_tiny",
    width_mult: float = 1.0,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown Sparse R-CNN variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]

    stem = scale_channels(int(spec["stem"]), float(width_mult), min_ch=16, divisor=8)
    feat = scale_channels(int(spec["feat"]), float(width_mult), min_ch=16, divisor=8)
    d_model = scale_channels(int(spec["d_model"]), float(width_mult), min_ch=32, divisor=8)
    return SparseRCNNDetector(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        stem_channels=int(stem),
        feat_channels=int(feat),
        backbone_depth=int(spec["depth"]),
        d_model=int(d_model),
        num_heads=int(spec["heads"]),
        num_queries=int(spec["q"]),
        stages=int(spec["stages"]),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 128, 128)
    m = build_sparse_rcnn_detector(in_channels=3, num_classes=2, variant="sparse_rcnn_tiny", width_mult=0.5)
    out = m(x)
    print("sparse_rcnn_tiny", len(out["class_logits"]), len(out["boxes"]))
    loss = sum(t.mean() for t in out["class_logits"]) + sum(t.mean() for t in out["boxes"])
    loss.backward()
    print("ok")

