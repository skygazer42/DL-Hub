from __future__ import annotations

import torch
from torch import nn

from ._common import StyleCodeEncoder, TinyDecoder, TinyEncoder
from .stylediffusion import StyleDiffusionDenoiser, _TimeMLP

_VARIANTS: dict[str, dict[str, int]] = {
    "sdedit_tiny": {"width": 24, "depth": 2},
    "sdedit_small": {"width": 32, "depth": 3},
    "sdedit_base": {"width": 48, "depth": 4},
}


class SDEditStyleTransfer(nn.Module):
    """SDEdit-style stylization (compact).

    Keeps the central SDEdit idea:
    - start from the encoded content image
    - perturb with controllable noise strength
    - iteratively denoise while steering toward a style reference latent
    """

    def __init__(
        self,
        *,
        in_channels: int,
        width: int,
        depth: int,
        style_dim: int = 64,
        steps: int = 6,
        strength: float = 0.4,
        edit_scale: float = 0.8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        c_in = int(in_channels)
        d = int(depth)
        self.encoder = TinyEncoder(
            in_channels=c_in,
            width=int(width),
            depth=d,
            dropout=float(dropout),
        )
        c_lat = int(self.encoder.out_channels)
        self.style_encoder = StyleCodeEncoder(
            in_channels=c_in,
            width=max(8, int(width) // 2),
            style_dim=int(style_dim),
        )
        self.time = _TimeMLP(int(style_dim))
        self.denoiser = StyleDiffusionDenoiser(
            channels=c_lat,
            width=max(8, c_lat // 2),
            depth=max(1, d),
            cond_dim=int(style_dim),
            dropout=float(dropout),
        )
        self.decoder = TinyDecoder(
            out_channels=c_in,
            in_channels=c_lat,
            depth=d,
            dropout=float(dropout),
        )
        self.steps = int(max(1, steps))
        self.strength = float(strength)
        self.edit_scale = float(edit_scale)

    def forward(self, content: torch.Tensor, style: torch.Tensor) -> dict[str, torch.Tensor]:
        x0 = self.encoder(content)
        target = self.encoder(style)
        style_code = self.style_encoder(style)

        noise = torch.randn_like(x0) * float(self.strength)
        x = x0 + noise
        bsz = int(x.shape[0])

        for i in range(int(self.steps)):
            t = torch.full((bsz, 1), 1.0 - float(i) / max(1, int(self.steps)), device=x.device)
            cond = style_code + self.time(t)
            eps = self.denoiser(x, cond)
            step = 0.6 / float(self.steps)
            drift = torch.tanh(target - x)
            x = x - float(step) * torch.tanh(eps) + float(self.edit_scale) * float(step) * drift
            x = x.clamp(-5.0, 5.0)

        y = self.decoder(x)
        return {
            "stylized": y,
            "edited_latent": x,
            "noise_strength": torch.tensor(float(self.strength), device=y.device),
            "edit_scale": torch.tensor(float(self.edit_scale), device=y.device),
        }


def build_sdedit_style_transfer(
    *,
    in_channels: int,
    image_size: int = 64,
    variant: str = "sdedit_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
    style_dim: int = 64,
    steps: int = 6,
    strength: float = 0.4,
    edit_scale: float = 0.8,
    ref_weight: float | None = None,
) -> nn.Module:
    _ = int(image_size)
    if ref_weight is not None:
        edit_scale = float(ref_weight)
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown SDEdit variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    cfg = _VARIANTS[name]
    width = max(8, int(int(cfg["width"]) * float(width_mult)))
    return SDEditStyleTransfer(
        in_channels=int(in_channels),
        width=width,
        depth=int(cfg["depth"]),
        style_dim=int(style_dim),
        steps=int(steps),
        strength=float(strength),
        edit_scale=float(edit_scale),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    s = torch.randn(2, 3, 64, 64)
    m = build_sdedit_style_transfer(
        in_channels=3,
        variant="sdedit_tiny",
        width_mult=0.5,
        steps=3,
        strength=0.3,
        edit_scale=0.7,
    )
    out = m(x, s)
    print("sdedit_tiny", tuple(out["stylized"].shape), tuple(out["edited_latent"].shape))
    loss = out["stylized"].mean() + out["edited_latent"].mean() + out["edit_scale"]
    loss.backward()
    print("ok")
