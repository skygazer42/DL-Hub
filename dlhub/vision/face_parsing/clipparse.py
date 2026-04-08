from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ._common import ParsingHead, TinyFaceEncoder, logits_to_parsing

_VARIANTS: dict[str, dict[str, int]] = {
    "clipparse_tiny": {"width": 16, "depth": 1},
    "clipparse_small": {"width": 24, "depth": 2},
    "clipparse_base": {"width": 32, "depth": 3},
}


class ClipparseFaceParser(nn.Module):
    """Semantic-gated multi-scale face parser."""

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
        self.num_classes = int(num_classes)
        self.encoder = TinyFaceEncoder(
            in_channels=int(in_channels),
            width=int(width),
            depth=int(depth),
            dropout=float(dropout),
        )
        c1, c2, c3 = (int(x) for x in self.encoder.out_channels)
        hidden = max(32, c3)
        self.class_bank = nn.Parameter(torch.randn(self.num_classes, c3) * 0.02)
        self.scale_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(c1 + c2 + c3, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 3, kernel_size=1, bias=True),
        )
        self.boundary_head = nn.Sequential(
            nn.Conv2d(c1 + c2, hidden, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, kernel_size=1, bias=True),
        )
        self.head = ParsingHead(
            in_channels=c1 + c2 + c3,
            hidden_channels=hidden,
            num_classes=int(num_classes),
            dropout=float(dropout),
        )

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        inp_hw = tuple(image.shape[-2:])
        c1, c2, c3 = self.encoder(image)
        c2_up = F.interpolate(c2, size=c1.shape[-2:], mode="bilinear", align_corners=False)
        c3_up = F.interpolate(c3, size=c1.shape[-2:], mode="bilinear", align_corners=False)
        fused = torch.cat([c1, c2_up, c3_up], dim=1)

        gate = torch.softmax(self.scale_gate(fused).flatten(1), dim=-1)
        mixed = torch.cat(
            [
                c1 * gate[:, 0].view(-1, 1, 1, 1),
                c2_up * gate[:, 1].view(-1, 1, 1, 1),
                c3_up * gate[:, 2].view(-1, 1, 1, 1),
            ],
            dim=1,
        )

        logits = self.head(mixed, out_hw=inp_hw)
        class_context = torch.einsum(
            "bkhw,kc->bchw",
            torch.softmax(logits, dim=1),
            self.class_bank.to(device=logits.device, dtype=logits.dtype),
        )
        class_bias = class_context.mean(dim=1, keepdim=True)
        boundary_feat = torch.cat([c1, c2_up], dim=1)
        boundary_map = torch.sigmoid(
            F.interpolate(self.boundary_head(boundary_feat), size=inp_hw, mode="bilinear", align_corners=False)
        )
        logits = logits + 0.1 * class_bias + 0.15 * boundary_map
        parsing_map = logits_to_parsing(logits)
        return {
            "logits": logits,
            "parsing_map": parsing_map,
            "scale_gate": gate,
            "boundary_map": boundary_map,
        }


def build_clipparse_face_parser(
    *,
    in_channels: int,
    num_classes: int,
    image_size: int = 64,
    variant: str = "clipparse_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    del image_size
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown SegFace variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    cfg = _VARIANTS[name]
    width = max(8, int(int(cfg["width"]) * float(width_mult)))
    return ClipparseFaceParser(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        width=width,
        depth=int(cfg["depth"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_clipparse_face_parser(
        in_channels=3,
        num_classes=11,
        variant="clipparse_tiny",
        width_mult=0.5,
    )
    out = m(x)
    print("clipparse_tiny", tuple(out["logits"].shape), tuple(out["boundary_map"].shape))
    loss = out["logits"].mean() + out["boundary_map"].mean()
    loss.backward()
    print("ok")

