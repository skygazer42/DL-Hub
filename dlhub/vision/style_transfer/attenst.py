from __future__ import annotations

import torch
from torch import nn

from ._common import TinyDecoder, TinyEncoder, channel_mean_std

_VARIANTS: dict[str, dict[str, int]] = {
    "attenst_tiny": {"width": 24, "depth": 2},
    "attenst_small": {"width": 32, "depth": 3},
    "attenst_base": {"width": 48, "depth": 4},
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


class ContentAwareAdaIN(nn.Module):
    def __init__(self, channels: int, *, eps: float = 1e-6) -> None:
        super().__init__()
        c = int(channels)
        if c <= 0:
            raise ValueError("channels must be > 0")
        self.gate = nn.Conv2d(c, c, kernel_size=1)
        self.eps = float(eps)

    def forward(
        self, content_feat: torch.Tensor, style_feat: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        c_mean, c_std = channel_mean_std(content_feat, eps=float(self.eps))
        s_mean, s_std = channel_mean_std(style_feat, eps=float(self.eps))
        gate = torch.sigmoid(self.gate(content_feat).mean(dim=(2, 3), keepdim=True))
        normed = (content_feat - c_mean) / c_std
        mix_mean = gate * s_mean + (1.0 - gate) * c_mean
        mix_std = gate * s_std + (1.0 - gate) * c_std
        return normed * mix_std + mix_mean, gate.mean()


class _AttenSTBlock(nn.Module):
    def __init__(self, *, dim: int, num_heads: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        d = int(dim)
        if d <= 0:
            raise ValueError("dim must be > 0")
        self.norm_q = nn.LayerNorm(d)
        self.norm_kv = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(
            d, int(num_heads), dropout=float(dropout), batch_first=True
        )
        self.norm_ff = nn.LayerNorm(d)
        self.ff = nn.Sequential(
            nn.Linear(d, max(d, d * 2)),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity(),
            nn.Linear(max(d, d * 2), d),
        )

    def forward(self, x_tokens: torch.Tensor, style_tokens: torch.Tensor) -> torch.Tensor:
        q = self.norm_q(x_tokens)
        kv = self.norm_kv(style_tokens)
        attn_out, _ = self.attn(q, kv, kv, need_weights=False)
        x = x_tokens + attn_out
        x = x + self.ff(self.norm_ff(x))
        return x


class AttenSTDenoiser(nn.Module):
    def __init__(
        self,
        *,
        channels: int,
        num_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        c = int(channels)
        self.time = _TimeToChannels(c)
        self.blocks = nn.ModuleList(
            [
                _AttenSTBlock(dim=c, num_heads=4, dropout=float(dropout))
                for _ in range(max(1, int(num_layers)))
            ]
        )
        self.out = nn.Conv2d(c, c, kernel_size=1)
        self.cadain = ContentAwareAdaIN(c)

    def forward(
        self, x: torch.Tensor, *, t: torch.Tensor, style_feat: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 4 or style_feat.ndim != 4:
            raise ValueError(
                f"Expected x/style_feat shapes (B, C, H, W), got {tuple(x.shape)} and {tuple(style_feat.shape)}"
            )
        b, c, h, w = x.shape
        x_tokens = x.to(torch.float32).flatten(2).transpose(1, 2)
        style_tokens = style_feat.to(torch.float32).flatten(2).transpose(1, 2)
        x_tokens = x_tokens + self.time(t).unsqueeze(1)
        for blk in self.blocks:
            x_tokens = blk(x_tokens, style_tokens)
        feat = x_tokens.transpose(1, 2).reshape(int(b), int(c), int(h), int(w))
        feat, gate_mean = self.cadain(feat, style_feat)
        return self.out(feat), gate_mean


class AttenSTStyleTransfer(nn.Module):
    """AttenST-style training-free diffusion stylization (toy).

    Uses attention-driven reference fusion followed by content-aware AdaIN at each denoising step.
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
        self.denoiser = AttenSTDenoiser(
            channels=c_lat,
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

    def forward(self, content: torch.Tensor, style: torch.Tensor) -> dict[str, torch.Tensor]:
        x0 = self.encoder(content)
        style_feat = self.encoder(style)
        x = x0 + torch.randn_like(x0) * float(self.strength)
        bsz = int(x.shape[0])
        gate_mean = torch.tensor(0.0, device=x.device)

        for i in range(int(self.steps)):
            t = torch.full((bsz, 1), 1.0 - float(i) / max(1, int(self.steps)), device=x.device)
            eps, gate_mean = self.denoiser(x, t=t, style_feat=style_feat)
            step = 0.6 / float(self.steps)
            x = x - float(step) * torch.tanh(eps)
            x = x.clamp(-5.0, 5.0)

        y = self.decoder(x)
        return {"stylized": y, "gate_mean": gate_mean}


def build_attenst_style_transfer(
    *,
    in_channels: int,
    image_size: int = 64,
    variant: str = "attenst_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
    steps: int = 6,
    strength: float = 0.5,
    num_layers: int = 2,
) -> nn.Module:
    _ = int(image_size)
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown AttenST variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    cfg = _VARIANTS[name]
    width = max(8, int(int(cfg["width"]) * float(width_mult)))
    return AttenSTStyleTransfer(
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
    m = build_attenst_style_transfer(
        in_channels=3,
        variant="attenst_tiny",
        width_mult=0.5,
        steps=3,
        strength=0.3,
        num_layers=1,
    )
    out = m(x, s)
    print("attenst_tiny", tuple(out["stylized"].shape), float(out["gate_mean"].item()))
    loss = out["stylized"].mean() + out["gate_mean"]
    loss.backward()
    print("ok")
