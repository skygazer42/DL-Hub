from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ._common import ParsingHead, TinyFaceEncoder, logits_to_parsing

_VARIANTS: dict[str, dict[str, int]] = {
    "mask_fpan_tiny": {"width": 16, "depth": 1},
    "mask_fpan_small": {"width": 24, "depth": 2},
    "mask_fpan_base": {"width": 32, "depth": 3},
}


class MaskFPANFaceParser(nn.Module):
    """Mask-FPAN style parser with de-occlusion and UV proxy branches."""

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
        self.deocc = nn.Sequential(
            nn.Conv2d(c3, hidden, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, c3, kernel_size=1, bias=True),
        )
        self.occlusion_head = nn.Sequential(
            nn.Conv2d(c1 + c2, hidden, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, kernel_size=1, bias=True),
        )
        self.uv_head = nn.Sequential(
            nn.Conv2d(c3, hidden, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 2, kernel_size=1, bias=True),
        )
        self.head = ParsingHead(
            in_channels=c3 * 2 + c2,
            hidden_channels=hidden,
            num_classes=int(num_classes),
            dropout=float(dropout),
        )

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        inp_hw = tuple(image.shape[-2:])
        c1, c2, c3 = self.encoder(image)
        c2_up = F.interpolate(c2, size=c1.shape[-2:], mode="bilinear", align_corners=False)
        occlusion_mask = torch.sigmoid(
            F.interpolate(
                self.occlusion_head(torch.cat([c1, c2_up], dim=1)),
                size=inp_hw,
                mode="bilinear",
                align_corners=False,
            )
        )
        uv_proxy = F.interpolate(self.uv_head(c3), size=inp_hw, mode="bilinear", align_corners=False)

        deocc_feat = c3 + self.deocc(c3)
        deocc_gate = 1.0 - F.interpolate(occlusion_mask, size=c3.shape[-2:], mode="bilinear", align_corners=False)
        fused_high = torch.cat([c3, deocc_feat * deocc_gate], dim=1)
        fused_high = torch.cat(
            [fused_high, F.interpolate(c2, size=c3.shape[-2:], mode="bilinear", align_corners=False)],
            dim=1,
        )
        logits = self.head(fused_high, out_hw=inp_hw)
        logits = logits + 0.1 * occlusion_mask + 0.05 * uv_proxy.mean(dim=1, keepdim=True)
        parsing_map = logits_to_parsing(logits)
        return {
            "logits": logits,
            "parsing_map": parsing_map,
            "occlusion_mask": occlusion_mask,
            "uv_proxy": uv_proxy,
        }


def build_mask_fpan_face_parser(
    *,
    in_channels: int,
    num_classes: int,
    image_size: int = 64,
    variant: str = "mask_fpan_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    del image_size
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown Mask-FPAN variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    cfg = _VARIANTS[name]
    width = max(8, int(int(cfg["width"]) * float(width_mult)))
    return MaskFPANFaceParser(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        width=width,
        depth=int(cfg["depth"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_mask_fpan_face_parser(
        in_channels=3,
        num_classes=11,
        variant="mask_fpan_tiny",
        width_mult=0.5,
    )
    out = m(x)
    print("mask_fpan_tiny", tuple(out["logits"].shape), tuple(out["uv_proxy"].shape))
    loss = out["logits"].mean() + out["occlusion_mask"].mean()
    loss.backward()
    print("ok")
