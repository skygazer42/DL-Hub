from __future__ import annotations

import torch
from torch import nn

from ._common import FiLM, ResBlock, StyleCodeEncoder, TinyDecoder, TinyEncoder, _conv_norm_act

_VARIANTS: dict[str, dict[str, int]] = {
    "stylediffusion_tiny": {"width": 24, "depth": 2},
    "stylediffusion_small": {"width": 32, "depth": 3},
    "stylediffusion_base": {"width": 48, "depth": 4},
}


class _TimeMLP(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        d = int(dim)
        if d <= 0:
            raise ValueError("dim must be > 0")
        self.net = nn.Sequential(
            nn.Linear(1, d),
            nn.ReLU(inplace=True),
            nn.Linear(d, d),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim != 2 or int(t.shape[1]) != 1:
            raise ValueError(f"t must have shape (B, 1), got {tuple(t.shape)}")
        return self.net(t.to(torch.float32))


class _CondResBlock(nn.Module):
    def __init__(self, *, channels: int, cond_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        c = int(channels)
        self.block = ResBlock(c, dropout=float(dropout))
        self.film = FiLM(channels=c, style_dim=int(cond_dim))

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return self.film(self.block(x), cond)


class StyleDiffusionDenoiser(nn.Module):
    def __init__(
        self,
        *,
        channels: int,
        width: int,
        depth: int,
        cond_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        c = int(channels)
        w = max(8, int(width))
        d = max(1, int(depth))
        self.in_proj = _conv_norm_act(c, w, kernel=3, stride=1, norm="gn")
        self.blocks = nn.ModuleList(
            [
                _CondResBlock(channels=w, cond_dim=int(cond_dim), dropout=float(dropout))
                for _ in range(d)
            ]
        )
        self.out_proj = nn.Conv2d(w, c, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x.to(torch.float32))
        for blk in self.blocks:
            h = blk(h, cond)
        return self.out_proj(h)


class StyleDiffusionStyleTransfer(nn.Module):
    """StyleDiffusion-style transfer (compact, latent diffusion img2img).

    This local family keeps the high-level mechanics used in diffusion img2img pipelines:
    - encode content to a latent feature map
    - inject noise with a controllable strength
    - iteratively denoise conditioned on a style reference embedding
    - decode back to image space

    It is not a pretrained Stable Diffusion pipeline; it is a tiny, CPU-friendly educational model.
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

    def _sample(
        self, x0: torch.Tensor, style_code: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bsz = int(x0.shape[0])
        noise = torch.randn_like(x0) * float(self.strength)
        x = x0 + noise

        for i in range(int(self.steps)):
            t = torch.full((bsz, 1), 1.0 - float(i) / max(1, int(self.steps)), device=x.device)
            cond = style_code + self.time(t)
            eps = self.denoiser(x, cond)
            step = 0.6 / float(self.steps)
            x = x - float(step) * torch.tanh(eps)
            x = x.clamp(-5.0, 5.0)
        return x, noise

    def forward(self, content: torch.Tensor, style: torch.Tensor) -> dict[str, torch.Tensor]:
        x0 = self.encoder(content)
        style_code = self.style_encoder(style)
        x, noise = self._sample(x0, style_code)
        y = self.decoder(x)
        return {
            "stylized": y,
            "latent": x,
            "noise_strength": torch.tensor(float(self.strength), device=y.device),
            "noise_mean_abs": noise.abs().mean(),
        }


def build_stylediffusion_style_transfer(
    *,
    in_channels: int,
    image_size: int = 64,
    variant: str = "stylediffusion_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
    style_dim: int = 64,
    steps: int = 6,
    strength: float = 0.5,
) -> nn.Module:
    _ = int(image_size)
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(
            f"Unknown StyleDiffusion variant: {variant!r}. Supported: {sorted(_VARIANTS)}"
        )
    cfg = _VARIANTS[name]
    width = max(8, int(int(cfg["width"]) * float(width_mult)))
    return StyleDiffusionStyleTransfer(
        in_channels=int(in_channels),
        width=width,
        depth=int(cfg["depth"]),
        style_dim=int(style_dim),
        steps=int(steps),
        strength=float(strength),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    s = torch.randn(2, 3, 64, 64)
    m = build_stylediffusion_style_transfer(
        in_channels=3, variant="stylediffusion_tiny", width_mult=0.5, steps=3, strength=0.3
    )
    out = m(x, s)
    print("stylediffusion_tiny", tuple(out["stylized"].shape), tuple(out["latent"].shape))
    loss = out["stylized"].mean() + out["latent"].mean() + out["noise_mean_abs"]
    loss.backward()
    print("ok")
