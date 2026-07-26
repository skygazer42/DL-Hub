from __future__ import annotations

import torch
from torch import nn

from ._common import PatchDiscriminator, StyleCodeEncoder, channel_mean_std, _conv_norm_act

_VARIANTS: dict[str, dict[str, int]] = {
    "ugatit_tiny": {"width": 24, "depth": 2},
    "ugatit_small": {"width": 32, "depth": 3},
    "ugatit_base": {"width": 48, "depth": 4},
}


class AdaLIN(nn.Module):
    """Adaptive Layer-Instance Normalization (toy).

    Mixes instance norm and layer norm, then applies a FiLM-like affine using a style code.
    """

    def __init__(self, *, channels: int, style_dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        c = int(channels)
        s = int(style_dim)
        if c <= 0 or s <= 0:
            raise ValueError("channels/style_dim must be > 0")
        self.rho = nn.Parameter(torch.full((1, c, 1, 1), 0.9))
        self.to_gamma = nn.Linear(s, c)
        self.to_beta = nn.Linear(s, c)
        self.eps = float(eps)

    def forward(self, x: torch.Tensor, style_code: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected x shape (B, C, H, W), got {tuple(x.shape)}")
        if style_code.ndim != 2:
            raise ValueError(f"Expected style_code shape (B, D), got {tuple(style_code.shape)}")

        in_mean, in_std = channel_mean_std(x, eps=float(self.eps))
        ln_mean = x.mean(dim=(1, 2, 3), keepdim=True)
        ln_var = x.var(dim=(1, 2, 3), keepdim=True, unbiased=False)
        ln_std = (ln_var + float(self.eps)).sqrt()

        x_in = (x - in_mean) / in_std
        x_ln = (x - ln_mean) / ln_std

        rho = self.rho.clamp(0.0, 1.0)
        y = rho * x_in + (1.0 - rho) * x_ln

        gamma = self.to_gamma(style_code).unsqueeze(-1).unsqueeze(-1)
        beta = self.to_beta(style_code).unsqueeze(-1).unsqueeze(-1)
        return y * (1.0 + gamma) + beta


class UGATITResBlock(nn.Module):
    def __init__(self, *, channels: int, style_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        c = int(channels)
        self.conv1 = nn.Conv2d(c, c, kernel_size=3, padding=1)
        self.norm1 = AdaLIN(channels=c, style_dim=int(style_dim))
        self.conv2 = nn.Conv2d(c, c, kernel_size=3, padding=1)
        self.norm2 = AdaLIN(channels=c, style_dim=int(style_dim))
        self.drop = nn.Dropout2d(float(dropout)) if float(dropout) > 0 else nn.Identity()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, style_code: torch.Tensor) -> torch.Tensor:
        y = self.conv1(x.to(torch.float32))
        y = self.norm1(y, style_code)
        y = self.act(y)
        y = self.drop(y)
        y = self.conv2(y)
        y = self.norm2(y, style_code)
        return self.act(x + y)


class UGATITGenerator(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        width: int,
        depth: int,
        style_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        c_in = int(in_channels)
        w = int(width)
        d = max(1, int(depth))

        self.down = nn.Sequential(
            _conv_norm_act(c_in, w, kernel=7, stride=1, norm="in"),
            _conv_norm_act(w, w * 2, kernel=3, stride=2, norm="in"),
            _conv_norm_act(w * 2, w * 4, kernel=3, stride=2, norm="in"),
        )
        cur = w * 4

        self.attn = nn.Conv2d(cur, 1, kernel_size=1)
        self.blocks = nn.ModuleList(
            [
                UGATITResBlock(channels=cur, style_dim=int(style_dim), dropout=float(dropout))
                for _ in range(d)
            ]
        )

        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            _conv_norm_act(cur, w * 2, kernel=3, stride=1, norm="in"),
            nn.Upsample(scale_factor=2, mode="nearest"),
            _conv_norm_act(w * 2, w, kernel=3, stride=1, norm="in"),
            nn.Conv2d(w, c_in, kernel_size=3, padding=1),
        )

    def forward(
        self, x: torch.Tensor, style_code: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.down(x.to(torch.float32))
        attn_map = torch.sigmoid(self.attn(feat))
        feat = feat * (1.0 + attn_map)
        for blk in self.blocks:
            feat = blk(feat, style_code)
        out = self.up(feat)
        return out, attn_map


class UGATITStyleTransfer(nn.Module):
    """U-GAT-IT style transfer (toy, reference-conditioned).

    The original U-GAT-IT is an unpaired translation method. This local family keeps the spirit:
    attention maps + adaptive normalization, but conditions on a style reference image.
    """

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
        c_in = int(in_channels)
        self.style_encoder = StyleCodeEncoder(
            in_channels=c_in, width=max(8, int(width) // 2), style_dim=int(style_dim)
        )
        self.g = UGATITGenerator(
            in_channels=c_in,
            width=int(width),
            depth=int(depth),
            style_dim=int(style_dim),
            dropout=float(dropout),
        )
        self.d = PatchDiscriminator(
            in_channels=c_in,
            width=max(8, int(width) // 2),
            depth=max(1, int(depth) - 1),
            dropout=float(dropout),
        )

    def forward(self, content: torch.Tensor, style: torch.Tensor) -> dict[str, torch.Tensor]:
        style_code = self.style_encoder(style)
        fake, attn_map = self.g(content, style_code)
        return {
            "stylized": fake,
            "attn_map": attn_map,
            "style_code": style_code,
            "logits_d_fake": self.d(fake.detach()),
            "logits_d_real": self.d(style.detach()),
        }


def build_ugatit_style_transfer(
    *,
    in_channels: int,
    image_size: int = 64,
    variant: str = "ugatit_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
    style_dim: int = 64,
) -> nn.Module:
    _ = int(image_size)
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown U-GAT-IT variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    cfg = _VARIANTS[name]
    width = max(8, int(int(cfg["width"]) * float(width_mult)))
    return UGATITStyleTransfer(
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
    m = build_ugatit_style_transfer(in_channels=3, variant="ugatit_tiny", width_mult=0.5)
    out = m(x, s)
    print("ugatit_tiny", tuple(out["stylized"].shape), tuple(out["attn_map"].shape))
    loss = out["stylized"].mean() + out["attn_map"].mean()
    loss.backward()
    print("ok")
