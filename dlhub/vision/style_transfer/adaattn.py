from __future__ import annotations

import math

import torch
from torch import nn

from ._common import TinyDecoder, TinyEncoder, channel_mean_std

_VARIANTS: dict[str, dict[str, int]] = {
    "adaattn_tiny": {"width": 24, "depth": 2},
    "adaattn_small": {"width": 32, "depth": 3},
    "adaattn_base": {"width": 48, "depth": 4},
}


class AdaAttNModule(nn.Module):
    """Adaptive Attention Normalization (toy).

    Computes per-position style mean/std from attention(content->style) and applies it to normalized
    content features. This keeps the core idea of AdaAttN while staying lightweight.
    """

    def __init__(self, channels: int, *, temperature: float = 1.0, eps: float = 1e-6) -> None:
        super().__init__()
        c = int(channels)
        if c <= 0:
            raise ValueError("channels must be > 0")
        self.to_q = nn.Conv2d(c, c, kernel_size=1, bias=False)
        self.to_k = nn.Conv2d(c, c, kernel_size=1, bias=False)
        self.to_v = nn.Conv2d(c, c, kernel_size=1, bias=False)
        self.temperature = float(temperature)
        self.eps = float(eps)

    def forward(self, content_feat: torch.Tensor, style_feat: torch.Tensor) -> torch.Tensor:
        if content_feat.ndim != 4 or style_feat.ndim != 4:
            raise ValueError(
                f"Expected content/style features (B, C, H, W), got {tuple(content_feat.shape)} and {tuple(style_feat.shape)}"
            )
        if content_feat.shape[1] != style_feat.shape[1]:
            raise ValueError("content/style channel mismatch")

        b, c, h, w = content_feat.shape
        q = self.to_q(content_feat).flatten(2).transpose(1, 2)  # (B, Nc, C)
        k = self.to_k(style_feat).flatten(2)  # (B, C, Ns)
        v = self.to_v(style_feat).flatten(2).transpose(1, 2)  # (B, Ns, C)

        temp = max(1e-3, float(self.temperature))
        logits = torch.bmm(q, k) / (math.sqrt(max(1.0, float(c))) * temp)  # (B, Nc, Ns)
        attn = torch.softmax(logits, dim=-1)

        mu = torch.bmm(attn, v)  # (B, Nc, C)
        m2 = torch.bmm(attn, v.pow(2))
        var = (m2 - mu.pow(2)).clamp_min(0.0)
        std = (var + float(self.eps)).sqrt()

        c_mean, c_std = channel_mean_std(content_feat, eps=float(self.eps))
        c_mean = c_mean.squeeze(-1).squeeze(-1).unsqueeze(1)  # (B, 1, C)
        c_std = c_std.squeeze(-1).squeeze(-1).unsqueeze(1)  # (B, 1, C)
        content_tokens = content_feat.flatten(2).transpose(1, 2)
        normed = (content_tokens - c_mean) / c_std
        out_tokens = normed * std + mu
        return out_tokens.transpose(1, 2).reshape(int(b), int(c), int(h), int(w))


class AdaAttNStyleTransfer(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        width: int,
        depth: int,
        dropout: float = 0.0,
        temperature: float = 1.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        c_in = int(in_channels)
        self.encoder = TinyEncoder(
            in_channels=c_in,
            width=int(width),
            depth=int(depth),
            dropout=float(dropout),
        )
        c_feat = int(self.encoder.out_channels)
        self.adaattn = AdaAttNModule(c_feat, temperature=float(temperature), eps=float(eps))
        self.decoder = TinyDecoder(
            out_channels=c_in,
            in_channels=c_feat,
            depth=int(depth),
            dropout=float(dropout),
        )

    def forward(self, content: torch.Tensor, style: torch.Tensor) -> dict[str, torch.Tensor]:
        f_c = self.encoder(content)
        f_s = self.encoder(style)
        t = self.adaattn(f_c, f_s)
        y = self.decoder(t)
        return {"stylized": y}


def build_adaattn_style_transfer(
    *,
    in_channels: int,
    image_size: int = 64,
    variant: str = "adaattn_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
    temperature: float = 1.0,
) -> nn.Module:
    _ = int(image_size)
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown AdaAttN variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    cfg = _VARIANTS[name]
    width = max(8, int(int(cfg["width"]) * float(width_mult)))
    return AdaAttNStyleTransfer(
        in_channels=int(in_channels),
        width=width,
        depth=int(cfg["depth"]),
        dropout=float(dropout),
        temperature=float(temperature),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    s = torch.randn(2, 3, 64, 64)
    m = build_adaattn_style_transfer(in_channels=3, variant="adaattn_tiny", width_mult=0.5)
    out = m(x, s)
    print("adaattn_tiny", tuple(out["stylized"].shape))
    loss = out["stylized"].mean()
    loss.backward()
    print("ok")
