from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from dlhub.vision.backbones._blocks import ConvBNAct

from ._common import ParsingHead, TinyFaceEncoder, check_nchw, logits_to_parsing

_VARIANTS: dict[str, dict[str, int]] = {
    "agrnet_tiny": {"width": 16, "depth": 1, "nodes": 8},
    "agrnet_small": {"width": 24, "depth": 2, "nodes": 11},
    "agrnet_base": {"width": 32, "depth": 3, "nodes": 14},
}


class AdaptiveGraphReasoning(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        dim = int(channels)
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.mix = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim),
        )

    def forward(self, nodes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        q = self.query(nodes)
        k = self.key(nodes)
        v = self.value(nodes)
        affinity = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(max(1, q.shape[-1]))
        affinity = torch.softmax(affinity, dim=-1)
        refined = torch.matmul(affinity, v)
        merged = self.mix(torch.cat([nodes, refined], dim=-1))
        return merged, affinity


class AGRNetFaceParser(nn.Module):
    """Adaptive graph parser with coarse parsing priors."""

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
        self.num_classes = int(num_classes)
        self.nodes = int(nodes)
        self.encoder = TinyFaceEncoder(
            in_channels=int(in_channels),
            width=int(width),
            depth=int(depth),
            dropout=float(dropout),
        )
        c1, c2, c3 = (int(x) for x in self.encoder.out_channels)
        hidden = max(32, c2)
        self.fuse = nn.Sequential(
            ConvBNAct(c1 + c2 + c3, hidden, kernel_size=1, stride=1, act="relu"),
            ConvBNAct(hidden, hidden, kernel_size=3, stride=1, act="relu"),
        )
        self.coarse_head = ParsingHead(
            in_channels=hidden,
            hidden_channels=hidden,
            num_classes=self.num_classes,
            dropout=float(dropout),
        )
        self.boundary_head = nn.Sequential(
            ConvBNAct(c1 + c2, hidden, kernel_size=3, stride=1, act="relu"),
            nn.Conv2d(hidden, 1, kernel_size=1, bias=True),
        )
        self.region_head = nn.Conv2d(hidden, self.nodes, kernel_size=1, bias=True)
        self.prior_head = nn.Conv2d(self.num_classes, self.nodes, kernel_size=1, bias=False)
        self.reason = AdaptiveGraphReasoning(hidden)
        self.refine = ConvBNAct(hidden * 2, hidden, kernel_size=3, stride=1, act="relu")
        self.refine_head = ParsingHead(
            in_channels=hidden,
            hidden_channels=hidden,
            num_classes=self.num_classes,
            dropout=float(dropout),
        )

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        image = check_nchw(image)
        inp_hw = tuple(image.shape[-2:])
        c1, c2, c3 = self.encoder(image)
        c2_up = F.interpolate(c2, size=c1.shape[-2:], mode="bilinear", align_corners=False)
        c3_up = F.interpolate(c3, size=c1.shape[-2:], mode="bilinear", align_corners=False)
        fused = self.fuse(torch.cat([c1, c2_up, c3_up], dim=1))

        coarse_low = self.coarse_head.net(fused)
        coarse_logits = F.interpolate(coarse_low, size=inp_hw, mode="bilinear", align_corners=False)
        coarse_prob = torch.softmax(coarse_low, dim=1)
        confidence = coarse_prob.max(dim=1, keepdim=True).values

        boundary_low = torch.sigmoid(self.boundary_head(torch.cat([c1, c2_up], dim=1)))
        boundary_out = F.interpolate(boundary_low, size=inp_hw, mode="bilinear", align_corners=False)

        assign_logits = self.region_head(fused) + self.prior_head(coarse_prob)
        assign_logits = assign_logits + 0.25 * boundary_low - 0.25 * (1.0 - confidence)
        assign = torch.softmax(assign_logits.flatten(2), dim=-1)
        feat_flat = fused.flatten(2)
        nodes = torch.einsum("bgn,bcn->bgc", assign, feat_flat)

        node_context = nodes.mean(dim=1, keepdim=True)
        nodes = nodes + 0.1 * node_context
        refined_nodes, affinity = self.reason(nodes)
        pixel_feat = torch.einsum("bgn,bgc->bcn", assign, refined_nodes).view_as(fused)
        refine_feat = self.refine(torch.cat([fused, pixel_feat], dim=1))
        logits = coarse_logits + self.refine_head(refine_feat, out_hw=inp_hw) + 0.1 * boundary_out

        parsing_map = logits_to_parsing(logits)
        return {
            "logits": logits,
            "parsing_map": parsing_map,
            "coarse_logits": coarse_logits,
            "boundary_map": boundary_out,
            "node_affinity": affinity,
        }


def build_agrnet_face_parser(
    *,
    in_channels: int,
    num_classes: int,
    image_size: int = 64,
    variant: str = "agrnet_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    del image_size
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown AGRNet variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    cfg = _VARIANTS[name]
    width = max(8, int(int(cfg["width"]) * float(width_mult)))
    return AGRNetFaceParser(
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
    m = build_agrnet_face_parser(
        in_channels=3,
        num_classes=11,
        variant="agrnet_tiny",
        width_mult=0.5,
    )
    out = m(x)
    print("agrnet_tiny", tuple(out["logits"].shape), tuple(out["boundary_map"].shape))
    loss = out["logits"].mean() + out["boundary_map"].mean()
    loss.backward()
    print("ok")
