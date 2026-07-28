from __future__ import annotations

import torch
from torch import nn

from ._common import TinyDecoder, TinyEncoder

_VARIANTS: dict[str, dict[str, int]] = {
    "ip_adapter_tiny": {"width": 24, "depth": 2},
    "ip_adapter_small": {"width": 32, "depth": 3},
    "ip_adapter_base": {"width": 48, "depth": 4},
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


class _CrossAttentionBlock(nn.Module):
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

        self.norm_q = nn.LayerNorm(d)
        self.norm_kv = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(
            embed_dim=d, num_heads=h, dropout=float(dropout), batch_first=True
        )

        hidden = max(d, int(d * float(mlp_ratio)))
        self.norm_ff = nn.LayerNorm(d)
        self.ff = nn.Sequential(
            nn.Linear(d, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity(),
            nn.Linear(hidden, d),
        )

    def forward(self, content_tokens: torch.Tensor, style_tokens: torch.Tensor) -> torch.Tensor:
        q = self.norm_q(content_tokens)
        kv = self.norm_kv(style_tokens)
        attn_out, _ = self.attn(q, kv, kv, need_weights=False)
        x = content_tokens + attn_out
        x = x + self.ff(self.norm_ff(x))
        return x


class IPAdapterDenoiser(nn.Module):
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
                _CrossAttentionBlock(
                    dim=c,
                    num_heads=int(num_heads),
                    mlp_ratio=2.0,
                    dropout=float(dropout),
                )
                for _ in range(max(1, int(num_layers)))
            ]
        )
        self.out = nn.Conv2d(c, c, kernel_size=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, style_feat: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or style_feat.ndim != 4:
            raise ValueError(
                f"Expected x/style_feat shapes (B, C, H, W), got {tuple(x.shape)} and {tuple(style_feat.shape)}"
            )
        b, c, h, w = x.shape
        if int(style_feat.shape[0]) != int(b) or int(style_feat.shape[1]) != int(c):
            raise ValueError("style feature shape mismatch")

        # Tokenize and apply cross-attention as an "image prompt adapter".
        content_tokens = x.to(torch.float32).flatten(2).transpose(1, 2)  # (B, N, C)
        style_tokens = style_feat.to(torch.float32).flatten(2).transpose(1, 2)  # (B, Ns, C)
        content_tokens = content_tokens + self.time(t).unsqueeze(1)
        for blk in self.blocks:
            content_tokens = blk(content_tokens, style_tokens)
        feat = content_tokens.transpose(1, 2).reshape(int(b), int(c), int(h), int(w))
        return self.out(feat)


class IPAdapterStyleTransfer(nn.Module):
    """IP-Adapter-style diffusion img2img (compact).

    Encodes a style reference image to feature tokens and uses cross-attention in latent-space
    denoising to inject the "image prompt" style signal.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        width: int,
        depth: int,
        steps: int = 6,
        strength: float = 0.5,
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
        self.denoiser = IPAdapterDenoiser(
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

    def forward(self, content: torch.Tensor, style: torch.Tensor) -> dict[str, torch.Tensor]:
        x0 = self.encoder(content)
        style_feat = self.encoder(style)
        noise = torch.randn_like(x0) * float(self.strength)
        x = x0 + noise
        bsz = int(x.shape[0])

        for i in range(int(self.steps)):
            t = torch.full((bsz, 1), 1.0 - float(i) / max(1, int(self.steps)), device=x.device)
            eps = self.denoiser(x, t, style_feat)
            step = 0.6 / float(self.steps)
            x = x - float(step) * torch.tanh(eps)
            x = x.clamp(-5.0, 5.0)

        y = self.decoder(x)
        return {"stylized": y}


def build_ip_adapter_style_transfer(
    *,
    in_channels: int,
    image_size: int = 64,
    variant: str = "ip_adapter_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
    steps: int = 6,
    strength: float = 0.5,
    num_layers: int = 2,
) -> nn.Module:
    _ = int(image_size)
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown IP-Adapter variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    cfg = _VARIANTS[name]
    width = max(8, int(int(cfg["width"]) * float(width_mult)))
    return IPAdapterStyleTransfer(
        in_channels=int(in_channels),
        width=width,
        depth=int(cfg["depth"]),
        steps=int(steps),
        strength=float(strength),
        num_layers=int(num_layers),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    s = torch.randn(2, 3, 64, 64)
    m = build_ip_adapter_style_transfer(
        in_channels=3,
        variant="ip_adapter_tiny",
        width_mult=0.5,
        steps=3,
        strength=0.3,
        num_layers=1,
    )
    out = m(x, s)
    print("ip_adapter_tiny", tuple(out["stylized"].shape))
    loss = out["stylized"].mean()
    loss.backward()
    print("ok")
