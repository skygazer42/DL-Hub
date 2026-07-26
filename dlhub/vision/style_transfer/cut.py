from __future__ import annotations

import torch
from torch import nn

from ._common import PatchDiscriminator, TinyEncoder, TinyUNet

_VARIANTS: dict[str, dict[str, int]] = {
    "cut_tiny": {"width": 24, "depth": 2},
    "cut_small": {"width": 32, "depth": 3},
    "cut_base": {"width": 48, "depth": 4},
}


class CUTStyleTransfer(nn.Module):
    """CUT-style unpaired translation (toy).

    This keeps only the high-level idea:
    - generator A->B
    - discriminator B
    - patch feature encoder for a contrastive-style signal (returned, not trained here)
    """

    def __init__(self, *, in_channels: int, width: int, depth: int, dropout: float = 0.0) -> None:
        super().__init__()
        c = int(in_channels)
        self.g = TinyUNet(in_channels=c, out_channels=c, width=int(width), depth=int(depth))
        self.d = PatchDiscriminator(
            in_channels=c,
            width=max(8, int(width) // 2),
            depth=max(1, int(depth) - 1),
            dropout=float(dropout),
        )
        self.f = TinyEncoder(
            in_channels=c,
            width=max(8, int(width) // 2),
            depth=max(1, int(depth) - 1),
            dropout=float(dropout),
        )

    def forward(
        self, content: torch.Tensor, style: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        _ = style
        fake = self.g(content)
        logits = self.d(fake)
        feat_in = self.f(content)
        feat_out = self.f(fake)
        v_in = feat_in.mean(dim=(2, 3))
        v_out = feat_out.mean(dim=(2, 3))
        sim = (v_in * v_out).sum(dim=1) / (v_in.norm(dim=1) * v_out.norm(dim=1)).clamp_min(1e-6)
        return {"stylized": fake, "fake_logits": logits, "contrastive_sim": sim}


def build_cut_style_transfer(
    *,
    in_channels: int,
    image_size: int = 64,
    variant: str = "cut_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    _ = int(image_size)
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown CUT variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    cfg = _VARIANTS[name]
    width = max(8, int(int(cfg["width"]) * float(width_mult)))
    return CUTStyleTransfer(
        in_channels=int(in_channels),
        width=width,
        depth=int(cfg["depth"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_cut_style_transfer(in_channels=3, variant="cut_tiny", width_mult=0.5)
    out = m(x, None)
    print("cut_tiny", tuple(out["stylized"].shape), tuple(out["fake_logits"].shape))
    loss = out["stylized"].mean() + out["fake_logits"].mean() + out["contrastive_sim"].mean()
    loss.backward()
    print("ok")
