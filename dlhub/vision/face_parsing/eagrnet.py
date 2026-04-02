from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from dlhub.vision.backbones._blocks import ConvBNAct

from ._common import ParsingHead, TinyFaceEncoder, check_nchw, logits_to_parsing

_VARIANTS: dict[str, dict[str, int]] = {
    "eagrnet_tiny": {"width": 16, "depth": 1, "nodes": 6},
    "eagrnet_small": {"width": 24, "depth": 2, "nodes": 8},
    "eagrnet_base": {"width": 32, "depth": 3, "nodes": 10},
}


class EdgeAwareGraphReasoning(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        dim = int(channels)
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)

    def forward(self, nodes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        q = self.query(nodes)
        k = self.key(nodes)
        v = self.value(nodes)
        affinity = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(max(1, q.shape[-1]))
        affinity = torch.softmax(affinity, dim=-1)
        refined = torch.matmul(affinity, v)
        return self.out(refined), affinity


class EAGRNetFaceParser(nn.Module):
    """Edge-aware graph reasoning parser."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        width: int,
        depth: int,
        nodes: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.encoder = TinyFaceEncoder(
            in_channels=int(in_channels),
            width=int(width),
            depth=int(depth),
            dropout=float(dropout),
        )
        c1, c2, c3 = (int(x) for x in self.encoder.out_channels)
        hidden = max(32, c2)
        self.nodes = int(nodes)
        self.fuse = nn.Sequential(
            ConvBNAct(c1 + c2 + c3, hidden, kernel_size=1, stride=1, act="relu"),
            ConvBNAct(hidden, hidden, kernel_size=3, stride=1, act="relu"),
        )
        self.edge_head = nn.Sequential(
            ConvBNAct(c1 + c2, hidden, kernel_size=3, stride=1, act="relu"),
            nn.Conv2d(hidden, 1, kernel_size=1, bias=True),
        )
        self.assign_head = nn.Conv2d(hidden + 1, self.nodes, kernel_size=1, bias=True)
        self.reason = EdgeAwareGraphReasoning(hidden)
        self.refine = ConvBNAct(hidden * 2, hidden, kernel_size=3, stride=1, act="relu")
        self.head = ParsingHead(
            in_channels=hidden,
            hidden_channels=hidden,
            num_classes=int(num_classes),
            dropout=float(dropout),
        )

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        image = check_nchw(image)
        inp_hw = tuple(image.shape[-2:])
        c1, c2, c3 = self.encoder(image)
        c2_up = F.interpolate(c2, size=c1.shape[-2:], mode="bilinear", align_corners=False)
        c3_up = F.interpolate(c3, size=c1.shape[-2:], mode="bilinear", align_corners=False)
        fused = self.fuse(torch.cat([c1, c2_up, c3_up], dim=1))

        edge_low = torch.sigmoid(self.edge_head(torch.cat([c1, c2_up], dim=1)))
        edge_out = F.interpolate(edge_low, size=inp_hw, mode="bilinear", align_corners=False)

        assign_logits = self.assign_head(torch.cat([fused, edge_low], dim=1))
        assign = torch.softmax(assign_logits.flatten(2), dim=-1)
        feat_flat = fused.flatten(2)
        edge_weight = 1.0 + edge_low.flatten(2)
        weighted_assign = assign * edge_weight
        weighted_assign = weighted_assign / weighted_assign.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        nodes = torch.einsum("bgn,bcn->bgc", weighted_assign, feat_flat)
        refined_nodes, affinity = self.reason(nodes)
        pixel_feat = torch.einsum("bgn,bgc->bcn", assign, refined_nodes).view_as(fused)
        refined = self.refine(torch.cat([fused, pixel_feat], dim=1))

        logits = self.head(refined, out_hw=inp_hw) + 0.15 * edge_out
        parsing_map = logits_to_parsing(logits)
        return {
            "logits": logits,
            "parsing_map": parsing_map,
            "edge_map": edge_out,
            "graph_attention": affinity,
        }


def build_eagrnet_face_parser(
    *,
    in_channels: int,
    num_classes: int,
    image_size: int = 64,
    variant: str = "eagrnet_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    del image_size
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown EAGRNet variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    cfg = _VARIANTS[name]
    width = max(8, int(int(cfg["width"]) * float(width_mult)))
    return EAGRNetFaceParser(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        width=width,
        depth=int(cfg["depth"]),
        nodes=int(cfg["nodes"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_eagrnet_face_parser(
        in_channels=3,
        num_classes=11,
        variant="eagrnet_tiny",
        width_mult=0.5,
    )
    out = m(x)
    print("eagrnet_tiny", tuple(out["logits"].shape), tuple(out["edge_map"].shape))
    loss = out["logits"].mean() + out["edge_map"].mean()
    loss.backward()
    print("ok")
