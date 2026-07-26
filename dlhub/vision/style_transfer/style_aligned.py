from __future__ import annotations

import torch
from torch import nn

from ._common import TinyDecoder, TinyEncoder

_VARIANTS: dict[str, dict[str, int]] = {
    "style_aligned_tiny": {"width": 24, "depth": 2},
    "style_aligned_small": {"width": 32, "depth": 3},
    "style_aligned_base": {"width": 48, "depth": 4},
}


class _TimeToChannels(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        c = int(channels)
        if c <= 0:
            raise ValueError("channels must be > 0")
        self.net = nn.Sequential(nn.Linear(1, c), nn.ReLU(inplace=True), nn.Linear(c, c))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim != 2 or int(t.shape[1]) != 1:
            raise ValueError(f"t must have shape (B, 1), got {tuple(t.shape)}")
        return self.net(t.to(torch.float32))


class _StyleAlignedBlock(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        d = int(dim)
        h = int(num_heads)
        if d <= 0 or h <= 0:
            raise ValueError("dim/num_heads must be > 0")
        self.norm1 = nn.LayerNorm(d)
        self.self_attn = nn.MultiheadAttention(d, h, dropout=float(dropout), batch_first=True)
        self.norm2 = nn.LayerNorm(d)
        self.ref_attn = nn.MultiheadAttention(d, h, dropout=float(dropout), batch_first=True)
        self.norm3 = nn.LayerNorm(d)
        hidden = max(d, int(d * float(mlp_ratio)))
        self.ff = nn.Sequential(
            nn.Linear(d, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity(),
            nn.Linear(hidden, d),
        )

    def forward(self, x: torch.Tensor, ref: torch.Tensor, *, ref_weight: float) -> torch.Tensor:
        # x/ref: (B, N, D)
        x1 = self.norm1(x)
        self_out, _ = self.self_attn(x1, x1, x1, need_weights=False)
        x = x + self_out

        x2 = self.norm2(x)
        ref = ref.to(x2.dtype)
        ref_out, _ = self.ref_attn(x2, ref, ref, need_weights=False)
        x = x + float(ref_weight) * ref_out

        x = x + self.ff(self.norm3(x))
        return x


class StyleAlignedDenoiser(nn.Module):
    def __init__(
        self,
        *,
        channels: int,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        c = int(channels)
        self.time = _TimeToChannels(c)
        self.blocks = nn.ModuleList(
            [
                _StyleAlignedBlock(
                    dim=c, num_heads=int(num_heads), mlp_ratio=2.0, dropout=float(dropout)
                )
                for _ in range(max(1, int(num_layers)))
            ]
        )
        self.out = nn.Conv2d(c, c, kernel_size=1)

    def forward(
        self,
        x: torch.Tensor,
        *,
        t: torch.Tensor,
        ref_feat: torch.Tensor,
        ref_weight: float,
    ) -> torch.Tensor:
        if x.ndim != 4 or ref_feat.ndim != 4:
            raise ValueError(
                f"Expected x/ref_feat (B, C, H, W), got {tuple(x.shape)} and {tuple(ref_feat.shape)}"
            )
        b, c, h, w = x.shape
        if tuple(ref_feat.shape) != (int(b), int(c), int(h), int(w)):
            raise ValueError("ref_feat shape mismatch")

        x_tokens = x.to(torch.float32).flatten(2).transpose(1, 2)
        ref_tokens = ref_feat.to(torch.float32).flatten(2).transpose(1, 2)
        x_tokens = x_tokens + self.time(t).unsqueeze(1)
        for blk in self.blocks:
            x_tokens = blk(x_tokens, ref_tokens, ref_weight=float(ref_weight))
        feat = x_tokens.transpose(1, 2).reshape(int(b), int(c), int(h), int(w))
        return self.out(feat)


class StyleAlignedStyleTransfer(nn.Module):
    """Style-aligned reference diffusion (toy).

    A diffusion-like img2img stylizer where denoising uses both:
    - self-attention over the evolving latent
    - reference-attention to a style latent (aligned K/V)
    """

    def __init__(
        self,
        *,
        in_channels: int,
        width: int,
        depth: int,
        steps: int = 6,
        strength: float = 0.5,
        ref_weight: float = 1.0,
        num_layers: int = 2,
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
        self.denoiser = StyleAlignedDenoiser(
            channels=c_lat,
            num_layers=int(num_layers),
            num_heads=4,
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
        self.ref_weight = float(ref_weight)

    def forward(self, content: torch.Tensor, style: torch.Tensor) -> dict[str, torch.Tensor]:
        x0 = self.encoder(content)
        ref = self.encoder(style)
        noise = torch.randn_like(x0) * float(self.strength)
        x = x0 + noise
        bsz = int(x.shape[0])

        for i in range(int(self.steps)):
            t = torch.full((bsz, 1), 1.0 - float(i) / max(1, int(self.steps)), device=x.device)
            eps = self.denoiser(x, t=t, ref_feat=ref, ref_weight=float(self.ref_weight))
            step = 0.6 / float(self.steps)
            x = x - float(step) * torch.tanh(eps)
            x = x.clamp(-5.0, 5.0)

        y = self.decoder(x)
        return {"stylized": y}


def build_style_aligned_style_transfer(
    *,
    in_channels: int,
    image_size: int = 64,
    variant: str = "style_aligned_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
    steps: int = 6,
    strength: float = 0.5,
    ref_weight: float = 1.0,
    num_layers: int = 2,
) -> nn.Module:
    _ = int(image_size)
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(
            f"Unknown StyleAligned variant: {variant!r}. Supported: {sorted(_VARIANTS)}"
        )
    cfg = _VARIANTS[name]
    width = max(8, int(int(cfg["width"]) * float(width_mult)))
    return StyleAlignedStyleTransfer(
        in_channels=int(in_channels),
        width=width,
        depth=int(cfg["depth"]),
        steps=int(steps),
        strength=float(strength),
        ref_weight=float(ref_weight),
        num_layers=int(num_layers),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    s = torch.randn(2, 3, 64, 64)
    m = build_style_aligned_style_transfer(
        in_channels=3,
        variant="style_aligned_tiny",
        width_mult=0.5,
        steps=3,
        strength=0.3,
        ref_weight=1.0,
        num_layers=1,
    )
    out = m(x, s)
    print("style_aligned_tiny", tuple(out["stylized"].shape))
    loss = out["stylized"].mean()
    loss.backward()
    print("ok")
