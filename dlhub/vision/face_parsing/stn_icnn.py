from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ._common import ParsingHead, TinyFaceEncoder, logits_to_parsing

_VARIANTS: dict[str, dict[str, int]] = {
    "stn_icnn_tiny": {"width": 16, "depth": 1},
    "stn_icnn_small": {"width": 24, "depth": 2},
    "stn_icnn_base": {"width": 32, "depth": 3},
}


class STNICNNFaceParser(nn.Module):
    """Spatial transformer + iterative CNN style face parser."""

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
        self.global_encoder = TinyFaceEncoder(
            in_channels=int(in_channels),
            width=int(width),
            depth=int(depth),
            dropout=float(dropout),
        )
        self.local_encoder = TinyFaceEncoder(
            in_channels=int(in_channels),
            width=int(width),
            depth=int(depth),
            dropout=float(dropout),
        )
        c2 = int(self.global_encoder.out_channels[1])
        c3 = int(self.global_encoder.out_channels[2])
        hidden = max(32, c3)
        self.locator = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(c3, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 6),
        )
        self.refine_gate = nn.Sequential(
            nn.Conv2d(c3 * 2, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, c3, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.head = ParsingHead(
            in_channels=c3 * 2 + c2,
            hidden_channels=hidden,
            num_classes=int(num_classes),
            dropout=float(dropout),
        )
        self._init_affine()

    def _init_affine(self) -> None:
        last = self.locator[-1]
        assert isinstance(last, nn.Linear)
        nn.init.zeros_(last.weight)
        with torch.no_grad():
            last.bias.copy_(torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=last.bias.dtype))

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        inp_hw = tuple(image.shape[-2:])
        _, g2, g3 = self.global_encoder(image)
        theta = self.locator(g3).view(-1, 2, 3)
        grid = F.affine_grid(theta, image.size(), align_corners=False)
        warped = F.grid_sample(
            image, grid, mode="bilinear", padding_mode="border", align_corners=False
        )

        _, _, l3 = self.local_encoder(warped)
        l3 = F.interpolate(l3, size=g3.shape[-2:], mode="bilinear", align_corners=False)
        refine_gate = self.refine_gate(torch.cat([g3, l3], dim=1))
        fused_local = l3 * refine_gate
        g2_up = F.interpolate(g2, size=g3.shape[-2:], mode="bilinear", align_corners=False)
        fused = torch.cat([g3, fused_local, g2_up], dim=1)

        logits = self.head(fused, out_hw=inp_hw)
        parsing_map = logits_to_parsing(logits)
        return {
            "logits": logits,
            "parsing_map": parsing_map,
            "affine_theta": theta,
            "refine_gate": refine_gate,
        }


def build_stn_icnn_face_parser(
    *,
    in_channels: int,
    num_classes: int,
    image_size: int = 64,
    variant: str = "stn_icnn_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    del image_size
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown STN-iCNN variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    cfg = _VARIANTS[name]
    width = max(8, int(int(cfg["width"]) * float(width_mult)))
    return STNICNNFaceParser(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        width=width,
        depth=int(cfg["depth"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_stn_icnn_face_parser(
        in_channels=3,
        num_classes=11,
        variant="stn_icnn_tiny",
        width_mult=0.5,
    )
    out = m(x)
    print("stn_icnn_tiny", tuple(out["logits"].shape), tuple(out["affine_theta"].shape))
    loss = out["logits"].mean() + out["refine_gate"].mean()
    loss.backward()
    print("ok")
