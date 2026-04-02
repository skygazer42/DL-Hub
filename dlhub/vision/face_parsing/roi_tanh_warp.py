from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ._common import ParsingHead, TinyFaceEncoder, logits_to_parsing, make_tanh_warp_grid

_VARIANTS: dict[str, dict[str, int | float]] = {
    "roi_tanh_warp_tiny": {"width": 16, "depth": 1, "focus": 1.5},
    "roi_tanh_warp_small": {"width": 24, "depth": 2, "focus": 1.8},
    "roi_tanh_warp_base": {"width": 32, "depth": 3, "focus": 2.0},
}


class RoITanhWarpFaceParser(nn.Module):
    """RoI Tanh-Warping style local-global face parser."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        width: int,
        depth: int,
        focus: float,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.focus = float(focus)
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
        self.roi_gate = nn.Sequential(
            nn.Conv2d(c3 * 2, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, kernel_size=1, bias=True),
        )
        self.head = ParsingHead(
            in_channels=c3 * 2 + c2,
            hidden_channels=hidden,
            num_classes=int(num_classes),
            dropout=float(dropout),
        )

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        inp_hw = tuple(image.shape[-2:])
        _, g2, g3 = self.global_encoder(image)

        grid = make_tanh_warp_grid(
            int(image.shape[0]),
            int(image.shape[-2]),
            int(image.shape[-1]),
            focus=float(self.focus),
            device=image.device,
            dtype=image.dtype,
        )
        warped = F.grid_sample(image, grid, mode="bilinear", padding_mode="border", align_corners=False)
        _, _, l3 = self.local_encoder(warped)
        l3 = F.interpolate(l3, size=g3.shape[-2:], mode="bilinear", align_corners=False)

        roi_attention = torch.sigmoid(self.roi_gate(torch.cat([g3, l3], dim=1)))
        local_enhanced = l3 * roi_attention
        g2_up = F.interpolate(g2, size=g3.shape[-2:], mode="bilinear", align_corners=False)
        fused = torch.cat([g3, local_enhanced, g2_up], dim=1)
        logits = self.head(fused, out_hw=inp_hw)
        parsing_map = logits_to_parsing(logits)
        return {"logits": logits, "parsing_map": parsing_map, "roi_attention": roi_attention}


def build_roi_tanh_warp_face_parser(
    *,
    in_channels: int,
    num_classes: int,
    image_size: int = 64,
    variant: str = "roi_tanh_warp_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    del image_size
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(
            f"Unknown RoI Tanh-Warp variant: {variant!r}. Supported: {sorted(_VARIANTS)}"
        )
    cfg = _VARIANTS[name]
    width = max(8, int(int(cfg["width"]) * float(width_mult)))
    return RoITanhWarpFaceParser(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        width=width,
        depth=int(cfg["depth"]),
        focus=float(cfg["focus"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_roi_tanh_warp_face_parser(
        in_channels=3,
        num_classes=11,
        variant="roi_tanh_warp_tiny",
        width_mult=0.5,
    )
    out = m(x)
    print("roi_tanh_warp_tiny", tuple(out["logits"].shape), tuple(out["roi_attention"].shape))
    loss = out["logits"].mean() + out["roi_attention"].mean()
    loss.backward()
    print("ok")
