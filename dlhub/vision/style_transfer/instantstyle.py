from __future__ import annotations

import torch
from torch import nn

from ._common import StyleCodeEncoder, TinyDecoder, TinyEncoder

_VARIANTS: dict[str, dict[str, int]] = {
    "instantstyle_tiny": {"width": 24, "depth": 2},
    "instantstyle_small": {"width": 32, "depth": 3},
    "instantstyle_base": {"width": 48, "depth": 4},
}


class _TimeToChannels(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        c = int(channels)
        if c <= 0:
            raise ValueError("channels must be > 0")
        self.net = nn.Sequential(
            nn.Linear(1, c),
            nn.ReLU(inplace=True),
            nn.Linear(c, c),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim != 2 or int(t.shape[1]) != 1:
            raise ValueError(f"t must have shape (B, 1), got {tuple(t.shape)}")
        return self.net(t.to(torch.float32))


class _InstantStyleBlock(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        style_dim: int,
        num_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        d = int(dim)
        s = int(style_dim)
        if d <= 0 or s <= 0:
            raise ValueError("dim/style_dim must be > 0")
        self.norm_q = nn.LayerNorm(d)
        self.norm_kv = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, int(num_heads), dropout=float(dropout), batch_first=True)
        self.to_gamma = nn.Linear(s, d)
        self.to_beta = nn.Linear(s, d)
        self.norm_ff = nn.LayerNorm(d)
        self.ff = nn.Sequential(
            nn.Linear(d, max(d, d * 2)),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity(),
            nn.Linear(max(d, d * 2), d),
        )

    def forward(
        self,
        content_tokens: torch.Tensor,
        style_tokens: torch.Tensor,
        style_code: torch.Tensor,
    ) -> torch.Tensor:
        q = self.norm_q(content_tokens)
        kv = self.norm_kv(style_tokens)
        attn_out, _ = self.attn(q, kv, kv, need_weights=False)
        x = content_tokens + attn_out
        gamma = self.to_gamma(style_code).unsqueeze(1)
        beta = self.to_beta(style_code).unsqueeze(1)
        x = x * (1.0 + gamma) + beta
        x = x + self.ff(self.norm_ff(x))
        return x


class InstantStyleDenoiser(nn.Module):
    def __init__(
        self,
        *,
        channels: int,
        style_dim: int,
        num_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        c = int(channels)
        self.time = _TimeToChannels(c)
        self.blocks = nn.ModuleList(
            [
                _InstantStyleBlock(
                    dim=c,
                    style_dim=int(style_dim),
                    num_heads=4,
                    dropout=float(dropout),
                )
                for _ in range(max(1, int(num_layers)))
            ]
        )
        self.out = nn.Conv2d(c, c, kernel_size=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, style_feat: torch.Tensor, style_code: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or style_feat.ndim != 4:
            raise ValueError(
                f"Expected x/style_feat shapes (B, C, H, W), got {tuple(x.shape)} and {tuple(style_feat.shape)}"
            )
        b, c, h, w = x.shape
        content_tokens = x.to(torch.float32).flatten(2).transpose(1, 2)
        style_tokens = style_feat.to(torch.float32).flatten(2).transpose(1, 2)
        content_tokens = content_tokens + self.time(t).unsqueeze(1)
        for blk in self.blocks:
            content_tokens = blk(content_tokens, style_tokens, style_code)
        feat = content_tokens.transpose(1, 2).reshape(int(b), int(c), int(h), int(w))
        return self.out(feat)


class InstantStyleStyleTransfer(nn.Module):
    """InstantStyle-style diffusion img2img (toy).

    The focus here is decoupled style injection:
    - style reference tokens provide local texture/layout style cues
    - a global style code modulates channels
    - a structure anchor pulls the latent back toward content geometry
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
        num_layers: int = 2,
        structure_weight: float = 0.8,
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
        self.denoiser = InstantStyleDenoiser(
            channels=c_lat,
            style_dim=int(style_dim),
            num_layers=int(num_layers),
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
        self.structure_weight = float(structure_weight)

    def forward(self, content: torch.Tensor, style: torch.Tensor) -> dict[str, torch.Tensor]:
        x0 = self.encoder(content)
        style_feat = self.encoder(style)
        style_code = self.style_encoder(style)
        noise = torch.randn_like(x0) * float(self.strength)
        x = x0 + noise
        bsz = int(x.shape[0])

        for i in range(int(self.steps)):
            t = torch.full((bsz, 1), 1.0 - float(i) / max(1, int(self.steps)), device=x.device)
            eps = self.denoiser(x, t, style_feat, style_code)
            step = 0.6 / float(self.steps)
            x = x - float(step) * torch.tanh(eps)
            x = x + float(self.structure_weight) * float(step) * torch.tanh(x0 - x)
            x = x.clamp(-5.0, 5.0)

        y = self.decoder(x)
        return {
            "stylized": y,
            "structure_weight": torch.tensor(float(self.structure_weight), device=y.device),
        }


def build_instantstyle_style_transfer(
    *,
    in_channels: int,
    image_size: int = 64,
    variant: str = "instantstyle_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
    style_dim: int = 64,
    steps: int = 6,
    strength: float = 0.5,
    num_layers: int = 2,
    ref_weight: float = 0.8,
) -> nn.Module:
    _ = int(image_size)
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(
            f"Unknown InstantStyle variant: {variant!r}. Supported: {sorted(_VARIANTS)}"
        )
    cfg = _VARIANTS[name]
    width = max(8, int(int(cfg["width"]) * float(width_mult)))
    return InstantStyleStyleTransfer(
        in_channels=int(in_channels),
        width=width,
        depth=int(cfg["depth"]),
        style_dim=int(style_dim),
        steps=int(steps),
        strength=float(strength),
        num_layers=int(num_layers),
        structure_weight=float(ref_weight),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    s = torch.randn(2, 3, 64, 64)
    m = build_instantstyle_style_transfer(
        in_channels=3,
        variant="instantstyle_tiny",
        width_mult=0.5,
        steps=3,
        strength=0.3,
        num_layers=1,
        ref_weight=0.7,
    )
    out = m(x, s)
    print("instantstyle_tiny", tuple(out["stylized"].shape), float(out["structure_weight"].item()))
    loss = out["stylized"].mean() + out["structure_weight"]
    loss.backward()
    print("ok")

