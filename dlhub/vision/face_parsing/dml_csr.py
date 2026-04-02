from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ._common import ParsingHead, TinyFaceEncoder, logits_to_parsing

_VARIANTS: dict[str, dict[str, int]] = {
    "dml_csr_tiny": {"width": 16, "depth": 1},
    "dml_csr_small": {"width": 24, "depth": 2},
    "dml_csr_base": {"width": 32, "depth": 3},
}


class DMLCSRFaceParser(nn.Module):
    """DML-CSR style face parser with decoupled edge refinement."""

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
        hidden = max(32, c3)
        self.coarse_head = ParsingHead(
            in_channels=c3 + c2,
            hidden_channels=hidden,
            num_classes=int(num_classes),
            dropout=float(dropout),
        )
        self.binary_edge_head = nn.Sequential(
            nn.Conv2d(c1 + c2, hidden, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, kernel_size=1, bias=True),
        )
        self.category_edge_head = nn.Sequential(
            nn.Conv2d(c1 + c2, hidden, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, int(num_classes), kernel_size=1, bias=True),
        )

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        inp_hw = tuple(image.shape[-2:])
        c1, c2, c3 = self.encoder(image)
        c2_up = F.interpolate(c2, size=c3.shape[-2:], mode="bilinear", align_corners=False)
        coarse_logits = self.coarse_head(torch.cat([c3, c2_up], dim=1), out_hw=inp_hw)

        edge_feat = torch.cat(
            [
                c1,
                F.interpolate(c2, size=c1.shape[-2:], mode="bilinear", align_corners=False),
            ],
            dim=1,
        )
        binary_edge = torch.sigmoid(
            F.interpolate(
                self.binary_edge_head(edge_feat),
                size=inp_hw,
                mode="bilinear",
                align_corners=False,
            )
        )
        category_edge = F.interpolate(
            self.category_edge_head(edge_feat),
            size=inp_hw,
            mode="bilinear",
            align_corners=False,
        )
        logits = coarse_logits + 0.2 * binary_edge + 0.1 * category_edge
        parsing_map = logits_to_parsing(logits)
        return {
            "logits": logits,
            "parsing_map": parsing_map,
            "binary_edge": binary_edge,
            "category_edge": category_edge,
        }


def build_dml_csr_face_parser(
    *,
    in_channels: int,
    num_classes: int,
    image_size: int = 64,
    variant: str = "dml_csr_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    del image_size
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown DML-CSR variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    cfg = _VARIANTS[name]
    width = max(8, int(int(cfg["width"]) * float(width_mult)))
    return DMLCSRFaceParser(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        width=width,
        depth=int(cfg["depth"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_dml_csr_face_parser(
        in_channels=3,
        num_classes=11,
        variant="dml_csr_tiny",
        width_mult=0.5,
    )
    out = m(x)
    print("dml_csr_tiny", tuple(out["logits"].shape), tuple(out["binary_edge"].shape))
    loss = out["logits"].mean() + out["binary_edge"].mean()
    loss.backward()
    print("ok")
