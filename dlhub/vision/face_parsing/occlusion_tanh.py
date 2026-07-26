from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ._common import ParsingHead, TinyFaceEncoder, logits_to_parsing, make_tanh_warp_grid

_VARIANTS: dict[str, dict[str, int | float]] = {
    "occlusion_tanh_tiny": {"width": 16, "depth": 1, "focus": 1.6},
    "occlusion_tanh_small": {"width": 24, "depth": 2, "focus": 1.9},
    "occlusion_tanh_base": {"width": 32, "depth": 3, "focus": 2.2},
}


class OcclusionTanhFaceParser(nn.Module):
    """Occlusion-aware parser with tanh-cartesian dual views."""

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
        self.base_encoder = TinyFaceEncoder(
            in_channels=int(in_channels),
            width=int(width),
            depth=int(depth),
            dropout=float(dropout),
        )
        self.warp_encoder = TinyFaceEncoder(
            in_channels=int(in_channels),
            width=int(width),
            depth=int(depth),
            dropout=float(dropout),
        )
        c2 = int(self.base_encoder.out_channels[1])
        c3 = int(self.base_encoder.out_channels[2])
        hidden = max(32, c3)
        self.occlusion_head = nn.Sequential(
            nn.Conv2d(c3 * 2, hidden, kernel_size=3, padding=1, bias=False),
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
        _, b2, b3 = self.base_encoder(image)

        grid = make_tanh_warp_grid(
            int(image.shape[0]),
            int(image.shape[-2]),
            int(image.shape[-1]),
            focus=float(self.focus),
            device=image.device,
            dtype=image.dtype,
        )
        warped = F.grid_sample(
            image, grid, mode="bilinear", padding_mode="border", align_corners=False
        )
        _, _, w3 = self.warp_encoder(warped)
        w3 = F.interpolate(w3, size=b3.shape[-2:], mode="bilinear", align_corners=False)

        occlusion_logits = self.occlusion_head(torch.cat([b3, w3], dim=1))
        occlusion_map = torch.sigmoid(
            F.interpolate(occlusion_logits, size=inp_hw, mode="bilinear", align_corners=False)
        )
        fusion_gate = torch.sigmoid(occlusion_logits)
        b2_up = F.interpolate(b2, size=b3.shape[-2:], mode="bilinear", align_corners=False)
        fused = torch.cat([b3, w3 * fusion_gate, b2_up], dim=1)
        logits = self.head(fused, out_hw=inp_hw)
        logits = logits + 0.15 * occlusion_map
        parsing_map = logits_to_parsing(logits)
        return {
            "logits": logits,
            "parsing_map": parsing_map,
            "occlusion_map": occlusion_map,
            "fusion_gate": fusion_gate,
        }


def build_occlusion_tanh_face_parser(
    *,
    in_channels: int,
    num_classes: int,
    image_size: int = 64,
    variant: str = "occlusion_tanh_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    del image_size
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(
            f"Unknown Occlusion-Tanh variant: {variant!r}. Supported: {sorted(_VARIANTS)}"
        )
    cfg = _VARIANTS[name]
    width = max(8, int(int(cfg["width"]) * float(width_mult)))
    return OcclusionTanhFaceParser(
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
    m = build_occlusion_tanh_face_parser(
        in_channels=3,
        num_classes=11,
        variant="occlusion_tanh_tiny",
        width_mult=0.5,
    )
    out = m(x)
    print("occlusion_tanh_tiny", tuple(out["logits"].shape), tuple(out["occlusion_map"].shape))
    loss = out["logits"].mean() + out["fusion_gate"].mean()
    loss.backward()
    print("ok")
