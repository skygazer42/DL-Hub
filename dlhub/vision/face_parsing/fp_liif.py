from __future__ import annotations

import torch
from torch import nn

from ._common import ParsingHead, TinyFaceEncoder, logits_to_parsing, make_coord_grid

_VARIANTS: dict[str, dict[str, int]] = {
    "fp_liif_tiny": {"width": 16, "depth": 1},
    "fp_liif_small": {"width": 24, "depth": 2},
    "fp_liif_base": {"width": 32, "depth": 3},
}


class FPLIIFFaceParser(nn.Module):
    """Local implicit image function face parser."""

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
        c2 = int(self.encoder.out_channels[1])
        c3 = int(self.encoder.out_channels[2])
        hidden = max(32, c3)
        self.coord_mlp = nn.Sequential(
            nn.Conv2d(c2 + c3 + 2, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.head = ParsingHead(
            in_channels=hidden,
            hidden_channels=hidden,
            num_classes=int(num_classes),
            dropout=float(dropout),
        )

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        inp_hw = tuple(image.shape[-2:])
        batch = int(image.shape[0])
        _, c2, c3 = self.encoder(image)
        c3_up = torch.nn.functional.interpolate(
            c3,
            size=c2.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        coords = make_coord_grid(
            batch,
            int(c2.shape[-2]),
            int(c2.shape[-1]),
            device=c2.device,
            dtype=c2.dtype,
        )
        implicit_feat = self.coord_mlp(torch.cat([c2, c3_up, coords], dim=1))
        logits = self.head(implicit_feat, out_hw=inp_hw)
        parsing_map = logits_to_parsing(logits)
        return {"logits": logits, "parsing_map": parsing_map, "implicit_features": implicit_feat}


def build_fp_liif_face_parser(
    *,
    in_channels: int,
    num_classes: int,
    image_size: int = 64,
    variant: str = "fp_liif_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    del image_size
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown FP-LIIF variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    cfg = _VARIANTS[name]
    width = max(8, int(int(cfg["width"]) * float(width_mult)))
    return FPLIIFFaceParser(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        width=width,
        depth=int(cfg["depth"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_fp_liif_face_parser(
        in_channels=3,
        num_classes=11,
        variant="fp_liif_tiny",
        width_mult=0.5,
    )
    out = m(x)
    print("fp_liif_tiny", tuple(out["logits"].shape), tuple(out["implicit_features"].shape))
    loss = out["logits"].mean() + out["implicit_features"].mean()
    loss.backward()
    print("ok")
