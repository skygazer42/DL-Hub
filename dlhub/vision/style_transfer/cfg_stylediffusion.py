from __future__ import annotations

import torch
from torch import nn

from ._common import StyleCodeEncoder, TinyDecoder, TinyEncoder
from .stylediffusion import StyleDiffusionDenoiser, _TimeMLP

_VARIANTS: dict[str, dict[str, int]] = {
    "cfg_stylediffusion_tiny": {"width": 24, "depth": 2},
    "cfg_stylediffusion_small": {"width": 32, "depth": 3},
    "cfg_stylediffusion_base": {"width": 48, "depth": 4},
}


class CFGStyleDiffusionStyleTransfer(nn.Module):
    """Classifier-Free Guidance (CFG) diffusion img2img style transfer (toy).

    Implements the key CFG formula used in Stable Diffusion ecosystems:
      eps = eps_uncond + s * (eps_cond - eps_uncond)

    Here, the conditioning signal is a style reference embedding.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        width: int,
        depth: int,
        style_dim: int = 64,
        steps: int = 6,
        strength: float = 0.5,
        guidance_scale: float = 2.0,
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
        self.guidance_scale = float(guidance_scale)

    def forward(self, content: torch.Tensor, style: torch.Tensor) -> dict[str, torch.Tensor]:
        x0 = self.encoder(content)
        style_code = self.style_encoder(style)

        noise = torch.randn_like(x0) * float(self.strength)
        x = x0 + noise

        bsz = int(x.shape[0])
        g = float(self.guidance_scale)
        for i in range(int(self.steps)):
            t = torch.full((bsz, 1), 1.0 - float(i) / max(1, int(self.steps)), device=x.device)
            time_code = self.time(t)
            cond = style_code + time_code
            uncond = time_code  # no style signal

            eps_cond = self.denoiser(x, cond)
            eps_uncond = self.denoiser(x, uncond)
            eps = eps_uncond + g * (eps_cond - eps_uncond)

            step = 0.6 / float(self.steps)
            x = x - float(step) * torch.tanh(eps)
            x = x.clamp(-5.0, 5.0)

        y = self.decoder(x)
        return {
            "stylized": y,
            "guidance_scale": torch.tensor(float(g), device=y.device),
        }


def build_cfg_stylediffusion_style_transfer(
    *,
    in_channels: int,
    image_size: int = 64,
    variant: str = "cfg_stylediffusion_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
    style_dim: int = 64,
    steps: int = 6,
    strength: float = 0.5,
    guidance_scale: float = 2.0,
) -> nn.Module:
    _ = int(image_size)
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(
            f"Unknown CFG StyleDiffusion variant: {variant!r}. Supported: {sorted(_VARIANTS)}"
        )
    cfg = _VARIANTS[name]
    width = max(8, int(int(cfg["width"]) * float(width_mult)))
    return CFGStyleDiffusionStyleTransfer(
        in_channels=int(in_channels),
        width=width,
        depth=int(cfg["depth"]),
        style_dim=int(style_dim),
        steps=int(steps),
        strength=float(strength),
        guidance_scale=float(guidance_scale),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    s = torch.randn(2, 3, 64, 64)
    m = build_cfg_stylediffusion_style_transfer(
        in_channels=3,
        variant="cfg_stylediffusion_tiny",
        width_mult=0.5,
        steps=3,
        strength=0.3,
        guidance_scale=2.0,
    )
    out = m(x, s)
    print(
        "cfg_stylediffusion_tiny", tuple(out["stylized"].shape), float(out["guidance_scale"].item())
    )
    loss = out["stylized"].mean() + out["guidance_scale"]
    loss.backward()
    print("ok")
