from __future__ import annotations

import torch
from torch import nn

from ._common import StyleCodeEncoder, TinyDecoder, TinyEncoder, _default_variants

_VARIANTS: dict[str, dict[str, int]] = _default_variants("artflow")


class FlowAffine(nn.Module):
    def __init__(self, *, channels: int, style_dim: int) -> None:
        super().__init__()
        c = int(channels)
        d = int(style_dim)
        if c <= 0 or d <= 0:
            raise ValueError("channels/style_dim must be > 0")
        self.to_scale = nn.Linear(d, c)
        self.to_shift = nn.Linear(d, c)

    def forward(self, feat: torch.Tensor, style_code: torch.Tensor) -> torch.Tensor:
        scale = self.to_scale(style_code).unsqueeze(-1).unsqueeze(-1)
        shift = self.to_shift(style_code).unsqueeze(-1).unsqueeze(-1)
        return feat * (1.0 + 0.25 * torch.tanh(scale)) + 0.25 * shift


class ArtFlowStyleTransfer(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        width: int,
        depth: int,
        style_dim: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.encoder = TinyEncoder(
            in_channels=int(in_channels),
            width=int(width),
            depth=int(depth),
            dropout=float(dropout),
        )
        c_feat = int(self.encoder.out_channels)
        self.style_encoder = StyleCodeEncoder(
            in_channels=int(in_channels),
            width=max(8, int(width) // 2),
            style_dim=int(style_dim),
        )
        self.flow = FlowAffine(channels=c_feat, style_dim=int(style_dim))
        self.decoder = TinyDecoder(
            out_channels=int(in_channels),
            in_channels=c_feat,
            depth=int(depth),
            dropout=float(dropout),
        )

    def forward(self, content: torch.Tensor, style: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.encoder(content)
        style_code = self.style_encoder(style)
        flowed = self.flow(feat, style_code)
        stylized = self.decoder(flowed)
        return {"stylized": stylized, "latent_norm": flowed.square().mean()}


def build_artflow_style_transfer(
    *,
    in_channels: int,
    image_size: int = 64,
    variant: str = "artflow_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
    style_dim: int = 64,
) -> nn.Module:
    _ = int(image_size)
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown ArtFlow variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    cfg = _VARIANTS[name]
    width = max(8, int(int(cfg["width"]) * float(width_mult)))
    return ArtFlowStyleTransfer(
        in_channels=int(in_channels),
        width=width,
        depth=int(cfg["depth"]),
        style_dim=int(style_dim),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    s = torch.randn(2, 3, 64, 64)
    m = build_artflow_style_transfer(in_channels=3, variant="artflow_tiny", width_mult=0.5)
    out = m(x, s)
    print("artflow_tiny", tuple(out["stylized"].shape))
    loss = out["stylized"].mean() + out["latent_norm"]
    loss.backward()
    print("ok")
