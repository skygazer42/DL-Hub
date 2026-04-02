from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn


def gram_matrix(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 4:
        raise ValueError(f"Expected x shape (B, C, H, W), got {tuple(x.shape)}")
    b, c, h, w = x.shape
    feat = x.view(int(b), int(c), int(h) * int(w))
    g = feat @ feat.transpose(1, 2)
    return g / max(1.0, float(c * h * w))


def channel_mean_std(x: torch.Tensor, *, eps: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor]:
    if x.ndim != 4:
        raise ValueError(f"Expected x shape (B, C, H, W), got {tuple(x.shape)}")
    mean = x.mean(dim=(2, 3), keepdim=True)
    var = x.var(dim=(2, 3), keepdim=True, unbiased=False)
    std = (var + float(eps)).sqrt()
    return mean, std


def adain(content: torch.Tensor, style: torch.Tensor, *, eps: float = 1e-6) -> torch.Tensor:
    c_mean, c_std = channel_mean_std(content, eps=float(eps))
    s_mean, s_std = channel_mean_std(style, eps=float(eps))
    normalized = (content - c_mean) / c_std
    return normalized * s_std + s_mean


def total_variation(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 4:
        raise ValueError(f"Expected x shape (B, C, H, W), got {tuple(x.shape)}")
    dy = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()
    dx = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
    return dx + dy


def wct(content: torch.Tensor, style: torch.Tensor, *, eps: float = 1e-5) -> torch.Tensor:
    """Whitening-color transform (toy implementation, per-sample)."""

    if content.shape != style.shape:
        raise ValueError(
            f"content/style shape mismatch: {tuple(content.shape)} vs {tuple(style.shape)}"
        )
    if content.ndim != 4:
        raise ValueError(f"Expected (B, C, H, W), got {tuple(content.shape)}")

    b, c, h, w = content.shape
    out = []
    for i in range(int(b)):
        fc = content[i].reshape(int(c), int(h) * int(w)).to(torch.float32)
        fs = style[i].reshape(int(c), int(h) * int(w)).to(torch.float32)

        mc = fc.mean(dim=1, keepdim=True)
        ms = fs.mean(dim=1, keepdim=True)
        xc = fc - mc
        xs = fs - ms

        cov_c = (xc @ xc.t()) / max(1, xc.shape[1] - 1)
        cov_s = (xs @ xs.t()) / max(1, xs.shape[1] - 1)
        cov_c = cov_c + torch.eye(int(c), device=cov_c.device) * float(eps)
        cov_s = cov_s + torch.eye(int(c), device=cov_s.device) * float(eps)

        evals_c, evecs_c = torch.linalg.eigh(cov_c)
        evals_s, evecs_s = torch.linalg.eigh(cov_s)

        evals_c = evals_c.clamp_min(float(eps))
        evals_s = evals_s.clamp_min(float(eps))

        wc = evecs_c @ torch.diag(evals_c.rsqrt()) @ evecs_c.t()
        cs = evecs_s @ torch.diag(evals_s.sqrt()) @ evecs_s.t()

        transformed = cs @ (wc @ xc) + ms
        out.append(transformed.reshape(int(c), int(h), int(w)))

    return torch.stack(out, dim=0).to(device=content.device, dtype=content.dtype)


def _conv_norm_act(
    in_ch: int,
    out_ch: int,
    *,
    kernel: int = 3,
    stride: int = 1,
    norm: str = "in",
    act: bool = True,
) -> nn.Sequential:
    padding = int(kernel) // 2
    layers: list[nn.Module] = [
        nn.Conv2d(
            int(in_ch),
            int(out_ch),
            int(kernel),
            stride=int(stride),
            padding=padding,
        )
    ]
    if str(norm) == "in":
        layers.append(nn.InstanceNorm2d(int(out_ch), affine=True))
    elif str(norm) == "gn":
        layers.append(nn.GroupNorm(num_groups=8, num_channels=int(out_ch)))
    elif str(norm) == "none":
        pass
    else:
        raise ValueError(f"Unknown norm: {norm!r}")
    if act:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class ResBlock(nn.Module):
    def __init__(self, channels: int, *, dropout: float = 0.0) -> None:
        super().__init__()
        c = int(channels)
        if c <= 0:
            raise ValueError("channels must be > 0")
        self.conv1 = _conv_norm_act(c, c, norm="in", act=True)
        self.drop = nn.Dropout2d(float(dropout)) if float(dropout) > 0 else nn.Identity()
        self.conv2 = _conv_norm_act(c, c, norm="in", act=False)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(x)
        y = self.drop(y)
        y = self.conv2(y)
        return self.act(x + y)


class TinyEncoder(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        width: int,
        depth: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        c_in = int(in_channels)
        w = int(width)
        if c_in <= 0:
            raise ValueError("in_channels must be > 0")
        if w < 8:
            raise ValueError("width must be >= 8")

        layers: list[nn.Module] = []
        layers.append(_conv_norm_act(c_in, w, kernel=7, stride=1, norm="in"))
        cur = w
        for _ in range(max(1, int(depth))):
            layers.append(_conv_norm_act(cur, cur * 2, kernel=3, stride=2, norm="in"))
            cur *= 2
            layers.append(ResBlock(cur, dropout=float(dropout)))
        self.net = nn.Sequential(*layers)
        self.out_channels = int(cur)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected input (B, C, H, W), got {tuple(x.shape)}")
        return self.net(x.to(torch.float32))


class TinyDecoder(nn.Module):
    def __init__(
        self,
        *,
        out_channels: int,
        in_channels: int,
        depth: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        c_out = int(out_channels)
        c_in = int(in_channels)
        if c_in <= 0:
            raise ValueError("in_channels must be > 0")
        if c_out <= 0:
            raise ValueError("out_channels must be > 0")

        layers: list[nn.Module] = []
        cur = int(c_in)
        for _ in range(max(1, int(depth))):
            layers.append(nn.Upsample(scale_factor=2, mode="nearest"))
            layers.append(_conv_norm_act(cur, max(8, cur // 2), kernel=3, stride=1, norm="in"))
            cur = max(8, cur // 2)
            layers.append(ResBlock(cur, dropout=float(dropout)))
        layers.append(nn.Conv2d(cur, c_out, kernel_size=3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected input (B, C, H, W), got {tuple(x.shape)}")
        return self.net(x.to(torch.float32))


class TinyUNet(nn.Module):
    def __init__(self, *, in_channels: int, out_channels: int, width: int, depth: int) -> None:
        super().__init__()
        c_in = int(in_channels)
        c_out = int(out_channels)
        w0 = int(width)
        d = max(2, int(depth))
        if c_in <= 0 or c_out <= 0:
            raise ValueError("channels must be > 0")
        if w0 < 8:
            raise ValueError("width must be >= 8")

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        cur = w0
        self.stem = _conv_norm_act(c_in, cur, kernel=3, stride=1, norm="in")

        for _ in range(d):
            self.downs.append(
                nn.Sequential(
                    ResBlock(cur),
                    _conv_norm_act(cur, cur * 2, stride=2, norm="in"),
                )
            )
            cur *= 2

        self.mid = nn.Sequential(ResBlock(cur), ResBlock(cur))

        for _ in range(d):
            self.ups.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="nearest"),
                    _conv_norm_act(cur, cur // 2, kernel=3, stride=1, norm="in"),
                    ResBlock(cur // 2),
                )
            )
            cur //= 2

        self.head = nn.Conv2d(cur, c_out, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        x = self.stem(x)
        for down in self.downs:
            x = down(x)
        x = self.mid(x)
        for up in self.ups:
            x = up(x)
        return self.head(x)


class PatchDiscriminator(nn.Module):
    def __init__(self, *, in_channels: int, width: int, depth: int, dropout: float = 0.0) -> None:
        super().__init__()
        c_in = int(in_channels)
        w0 = int(width)
        if c_in <= 0:
            raise ValueError("in_channels must be > 0")
        if w0 < 8:
            raise ValueError("width must be >= 8")

        layers: list[nn.Module] = []
        cur = w0
        layers.append(nn.Conv2d(c_in, cur, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        for _ in range(max(1, int(depth))):
            nxt = min(512, cur * 2)
            layers.append(nn.Conv2d(cur, nxt, kernel_size=4, stride=2, padding=1))
            layers.append(nn.InstanceNorm2d(nxt, affine=True))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            if float(dropout) > 0:
                layers.append(nn.Dropout2d(float(dropout)))
            cur = nxt
        layers.append(nn.Conv2d(cur, 1, kernel_size=3, stride=1, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected (B, C, H, W), got {tuple(x.shape)}")
        return self.net(x.to(torch.float32))


class FiLM(nn.Module):
    def __init__(self, *, channels: int, style_dim: int) -> None:
        super().__init__()
        c = int(channels)
        s = int(style_dim)
        if c <= 0 or s <= 0:
            raise ValueError("channels/style_dim must be > 0")
        self.to_gamma = nn.Linear(s, c)
        self.to_beta = nn.Linear(s, c)

    def forward(self, x: torch.Tensor, style_code: torch.Tensor) -> torch.Tensor:
        if style_code.ndim != 2:
            raise ValueError(f"style_code must have shape (B, D), got {tuple(style_code.shape)}")
        gamma = self.to_gamma(style_code).unsqueeze(-1).unsqueeze(-1)
        beta = self.to_beta(style_code).unsqueeze(-1).unsqueeze(-1)
        return x * (1.0 + gamma) + beta


class StyleCodeEncoder(nn.Module):
    def __init__(self, *, in_channels: int, width: int, style_dim: int) -> None:
        super().__init__()
        w = int(width)
        self.enc = nn.Sequential(
            _conv_norm_act(int(in_channels), w, kernel=3, stride=2, norm="in"),
            _conv_norm_act(w, w * 2, kernel=3, stride=2, norm="in"),
            _conv_norm_act(w * 2, w * 2, kernel=3, stride=2, norm="in"),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Linear(w * 2, int(style_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.enc(x.to(torch.float32))
        x = self.pool(x).flatten(1)
        return self.proj(x)


class SpatialCrossAttention(nn.Module):
    def __init__(self, *, channels: int, temperature: float = 1.0) -> None:
        super().__init__()
        c = int(channels)
        if c <= 0:
            raise ValueError("channels must be > 0")
        self.to_q = nn.Conv2d(c, c, kernel_size=1, bias=False)
        self.to_k = nn.Conv2d(c, c, kernel_size=1, bias=False)
        self.to_v = nn.Conv2d(c, c, kernel_size=1, bias=False)
        self.proj = nn.Conv2d(c, c, kernel_size=1)
        self.temperature = max(1e-6, float(temperature))

    def forward(self, content_feat: torch.Tensor, style_feat: torch.Tensor) -> torch.Tensor:
        if content_feat.ndim != 4 or style_feat.ndim != 4:
            raise ValueError(
                "Expected content/style features with shape (B, C, H, W), "
                f"got {tuple(content_feat.shape)} and {tuple(style_feat.shape)}"
            )
        b, c, h, w = content_feat.shape
        q = self.to_q(content_feat).flatten(2).transpose(1, 2)
        k = self.to_k(style_feat).flatten(2)
        v = self.to_v(style_feat).flatten(2).transpose(1, 2)
        scale = math.sqrt(max(1.0, float(c))) * float(self.temperature)
        attn = torch.softmax(torch.bmm(q, k) / scale, dim=-1)
        out = torch.bmm(attn, v).transpose(1, 2).reshape(int(b), int(c), int(h), int(w))
        return self.proj(out)


class TinyResNetGenerator(nn.Module):
    def __init__(self, *, in_channels: int, width: int, depth: int, dropout: float = 0.0) -> None:
        super().__init__()
        c = int(in_channels)
        w = int(width)
        d = max(1, int(depth))
        if c <= 0 or w < 8:
            raise ValueError("in_channels must be > 0 and width must be >= 8")
        self.stem = nn.Sequential(
            _conv_norm_act(c, w, kernel=7, stride=1, norm="in"),
            _conv_norm_act(w, w * 2, kernel=3, stride=2, norm="in"),
            _conv_norm_act(w * 2, w * 4, kernel=3, stride=2, norm="in"),
        )
        cur = w * 4
        self.blocks = nn.Sequential(*[ResBlock(cur, dropout=float(dropout)) for _ in range(d)])
        self.head = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            _conv_norm_act(cur, w * 2, kernel=3, stride=1, norm="in"),
            nn.Upsample(scale_factor=2, mode="nearest"),
            _conv_norm_act(w * 2, w, kernel=3, stride=1, norm="in"),
            nn.Conv2d(w, c, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected input (B, C, H, W), got {tuple(x.shape)}")
        x = x.to(torch.float32)
        return self.head(self.blocks(self.stem(x)))


@dataclass(frozen=True)
class VariantSpec:
    width: int
    depth: int


def _default_variants(prefix: str) -> dict[str, dict[str, int]]:
    p = str(prefix).strip()
    return {
        f"{p}_tiny": {"width": 24, "depth": 2},
        f"{p}_small": {"width": 32, "depth": 3},
        f"{p}_base": {"width": 48, "depth": 4},
    }
