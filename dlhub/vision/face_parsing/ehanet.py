from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from dlhub.vision.backbones._blocks import ConvBNAct

from ._common import ParsingHead, TinyFaceEncoder, check_nchw, logits_to_parsing

_VARIANTS: dict[str, dict[str, int]] = {
    "ehanet_tiny": {"width": 16, "depth": 1},
    "ehanet_small": {"width": 24, "depth": 2},
    "ehanet_base": {"width": 32, "depth": 3},
}


class StageContextAttention(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        dim = int(channels)
        hidden = max(8, dim // 4)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(dim, hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, dim, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        attn = self.net(x)
        return x * attn, attn


class SemanticGapCompensation(nn.Module):
    def __init__(self, low_channels: int, high_channels: int, out_channels: int) -> None:
        super().__init__()
        out_ch = int(out_channels)
        self.low_proj = ConvBNAct(int(low_channels), out_ch, kernel_size=1, stride=1, act="relu")
        self.high_proj = ConvBNAct(int(high_channels), out_ch, kernel_size=1, stride=1, act="relu")
        self.gate = nn.Sequential(
            nn.Conv2d(out_ch * 2, out_ch, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.mix = ConvBNAct(out_ch * 2, out_ch, kernel_size=3, stride=1, act="relu")

    def forward(self, low: torch.Tensor, high: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        low_feat = self.low_proj(low)
        high_feat = F.interpolate(
            self.high_proj(high),
            size=low.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        gate = self.gate(torch.cat([low_feat, high_feat], dim=1))
        fused = self.mix(torch.cat([low_feat, high_feat * gate], dim=1))
        return fused, gate


class EHANetFaceParser(nn.Module):
    """Hierarchical aggregation parser inspired by EHANet."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        width: int,
        depth: int,
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
        self.attn1 = StageContextAttention(c1)
        self.attn2 = StageContextAttention(c2)
        self.attn3 = StageContextAttention(c3)
        self.p3 = ConvBNAct(c3, hidden, kernel_size=1, stride=1, act="relu")
        self.sgcb23 = SemanticGapCompensation(c2, hidden, hidden)
        self.sgcb12 = SemanticGapCompensation(c1, hidden, hidden)
        self.boundary_head = nn.Sequential(
            ConvBNAct(hidden, hidden, kernel_size=3, stride=1, act="relu"),
            nn.Conv2d(hidden, 1, kernel_size=1, bias=True),
        )
        self.head = ParsingHead(
            in_channels=hidden * 3,
            hidden_channels=hidden,
            num_classes=int(num_classes),
            dropout=float(dropout),
        )

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        image = check_nchw(image)
        inp_hw = tuple(image.shape[-2:])
        c1, c2, c3 = self.encoder(image)
        a1, w1 = self.attn1(c1)
        a2, w2 = self.attn2(c2)
        a3, w3 = self.attn3(c3)

        p3 = self.p3(a3)
        p2, gate23 = self.sgcb23(a2, p3)
        p1, gate12 = self.sgcb12(a1, p2)

        p2_up = F.interpolate(p2, size=p1.shape[-2:], mode="bilinear", align_corners=False)
        p3_up = F.interpolate(p3, size=p1.shape[-2:], mode="bilinear", align_corners=False)
        fused = torch.cat([p1, p2_up, p3_up], dim=1)

        boundary_out = torch.sigmoid(
            F.interpolate(self.boundary_head(p1), size=inp_hw, mode="bilinear", align_corners=False)
        )
        logits = self.head(fused, out_hw=inp_hw) + 0.15 * boundary_out
        parsing_map = logits_to_parsing(logits)
        return {
            "logits": logits,
            "parsing_map": parsing_map,
            "boundary_map": boundary_out,
            "stage_attention": (w1, w2, w3),
            "fusion_gates": (gate12, gate23),
        }


def build_ehanet_face_parser(
    *,
    in_channels: int,
    num_classes: int,
    image_size: int = 64,
    variant: str = "ehanet_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    del image_size
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown EHANet variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    cfg = _VARIANTS[name]
    width = max(8, int(int(cfg["width"]) * float(width_mult)))
    return EHANetFaceParser(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        width=width,
        depth=int(cfg["depth"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_ehanet_face_parser(
        in_channels=3,
        num_classes=11,
        variant="ehanet_tiny",
        width_mult=0.5,
    )
    out = m(x)
    print("ehanet_tiny", tuple(out["logits"].shape), tuple(out["boundary_map"].shape))
    loss = out["logits"].mean() + out["boundary_map"].mean()
    loss.backward()
    print("ok")
