from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from ._common import StyleCodeEncoder, TinyDecoder, TinyEncoder
from .stylediffusion import StyleDiffusionDenoiser, _TimeMLP

_VARIANTS: dict[str, dict[str, int]] = {
    "controlnet_tiny": {"width": 24, "depth": 2},
    "controlnet_small": {"width": 32, "depth": 3},
    "controlnet_base": {"width": 48, "depth": 4},
}


def _sobel_edges(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 4:
        raise ValueError(f"Expected x shape (B, C, H, W), got {tuple(x.shape)}")
    gray = x.to(torch.float32).mean(dim=1, keepdim=True)

    kx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], device=gray.device)
    ky = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], device=gray.device)
    kx = kx.view(1, 1, 3, 3)
    ky = ky.view(1, 1, 3, 3)

    gx = F.conv2d(gray, kx, padding=1)
    gy = F.conv2d(gray, ky, padding=1)
    edges = (gx.pow(2) + gy.pow(2) + 1e-6).sqrt()
    return torch.tanh(edges)


class ControlNetStyleTransfer(nn.Module):
    """ControlNet-style diffusion img2img (toy).

    Approximates the ControlNet idea:
    - compute a structural "hint" (edges) from the content image
    - encode hint to the latent space and feed it as additional conditioning during denoising
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
        control_scale: float = 1.0,
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

        self.hint_encoder = TinyEncoder(
            in_channels=1,
            width=int(width),
            depth=d,
            dropout=float(dropout),
        )
        if int(self.hint_encoder.out_channels) != int(c_lat):
            raise RuntimeError("hint/content latent channel mismatch; check encoder configurations")

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
        self.control_scale = float(control_scale)

    def forward(self, content: torch.Tensor, style: torch.Tensor) -> dict[str, torch.Tensor]:
        x0 = self.encoder(content)
        style_code = self.style_encoder(style)

        hint = _sobel_edges(content)
        hint_latent = self.hint_encoder(hint)

        noise = torch.randn_like(x0) * float(self.strength)
        x = x0 + noise

        bsz = int(x.shape[0])
        for i in range(int(self.steps)):
            t = torch.full((bsz, 1), 1.0 - float(i) / max(1, int(self.steps)), device=x.device)
            cond = style_code + self.time(t)
            eps = self.denoiser(x + float(self.control_scale) * hint_latent, cond)
            step = 0.6 / float(self.steps)
            x = x - float(step) * torch.tanh(eps)
            x = x.clamp(-5.0, 5.0)

        y = self.decoder(x)
        return {
            "stylized": y,
            "hint": hint,
            "hint_latent_mean_abs": hint_latent.abs().mean(),
        }


def build_controlnet_style_transfer(
    *,
    in_channels: int,
    image_size: int = 64,
    variant: str = "controlnet_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
    style_dim: int = 64,
    steps: int = 6,
    strength: float = 0.5,
    control_scale: float = 1.0,
) -> nn.Module:
    _ = int(image_size)
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown ControlNet variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    cfg = _VARIANTS[name]
    width = max(8, int(int(cfg["width"]) * float(width_mult)))
    return ControlNetStyleTransfer(
        in_channels=int(in_channels),
        width=width,
        depth=int(cfg["depth"]),
        style_dim=int(style_dim),
        steps=int(steps),
        strength=float(strength),
        control_scale=float(control_scale),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    s = torch.randn(2, 3, 64, 64)
    m = build_controlnet_style_transfer(
        in_channels=3, variant="controlnet_tiny", width_mult=0.5, steps=3, strength=0.3
    )
    out = m(x, s)
    print("controlnet_tiny", tuple(out["stylized"].shape), tuple(out["hint"].shape))
    loss = out["stylized"].mean() + out["hint_latent_mean_abs"]
    loss.backward()
    print("ok")

