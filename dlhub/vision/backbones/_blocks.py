from __future__ import annotations

import math

import torch
from torch import nn


def make_divisible(v: int, divisor: int = 8) -> int:
    d = int(divisor)
    if d <= 0:
        raise ValueError("divisor must be > 0")
    x = int(v)
    if x <= 0:
        return d
    return int((x + d - 1) // d * d)


def scale_channels(
    ch: int,
    width_mult: float,
    *,
    min_ch: int = 8,
    divisor: int = 8,
) -> int:
    v = max(int(min_ch), int(round(int(ch) * float(width_mult))))
    return make_divisible(v, int(divisor))


class ConvBNAct(nn.Sequential):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        *,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int | None = None,
        groups: int = 1,
        act: str = "relu",
    ) -> None:
        k = int(kernel_size)
        if padding is None:
            padding = k // 2

        act_name = str(act).lower().strip()
        if act_name in {"relu", "relu_"}:
            act_layer: nn.Module = nn.ReLU(inplace=True)
        elif act_name in {"relu6"}:
            act_layer = nn.ReLU6(inplace=True)
        elif act_name in {"gelu"}:
            act_layer = nn.GELU()
        elif act_name in {"silu", "swish"}:
            act_layer = nn.SiLU(inplace=True)
        elif act_name in {"hswish", "hardswish"}:
            act_layer = nn.Hardswish(inplace=True)
        elif act_name in {"leaky", "leakyrelu"}:
            act_layer = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        elif act_name in {"mish"}:
            act_layer = nn.Mish()
        else:
            raise ValueError(f"Unknown activation: {act!r}")

        super().__init__(
            nn.Conv2d(
                int(in_ch),
                int(out_ch),
                kernel_size=int(k),
                stride=int(stride),
                padding=int(padding),
                groups=int(groups),
                bias=False,
            ),
            nn.BatchNorm2d(int(out_ch)),
            act_layer,
        )


class DropPath(nn.Module):
    """Stochastic depth / DropPath (per-sample).

    A tiny, dependency-free implementation. Use only during training.
    """

    def __init__(self, p: float = 0.0) -> None:
        super().__init__()
        self.p = float(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p <= 0.0:
            return x
        keep = 1.0 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.empty(shape, device=x.device, dtype=x.dtype).bernoulli_(keep)
        return x * mask / keep


class SqueezeExcite(nn.Module):
    def __init__(self, channels: int, *, se_ratio: float = 0.25) -> None:
        super().__init__()
        c = int(channels)
        hidden = max(8, int(round(c * float(se_ratio))))
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Conv2d(c, hidden, kernel_size=1)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden, c, kernel_size=1)
        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.pool(x)
        s = self.fc1(s)
        s = self.act(s)
        s = self.fc2(s)
        s = self.gate(s)
        return x * s


class ECALayer(nn.Module):
    """Efficient Channel Attention (ECA).

    Global average pool -> 1D conv over channels -> sigmoid.
    """

    def __init__(self, channels: int, *, kernel_size: int = 3) -> None:
        super().__init__()
        c = int(channels)
        k = int(kernel_size)
        if c <= 0:
            raise ValueError("channels must be > 0")
        if k <= 0 or k % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.pool(x)  # (B, C, 1, 1)
        y = y.squeeze(-1).transpose(1, 2)  # (B, 1, C)
        y = self.conv(y)
        y = self.gate(y).transpose(1, 2).unsqueeze(-1)  # (B, C, 1, 1)
        return x * y


class CBAM(nn.Module):
    """Convolutional Block Attention Module (CBAM).

    This is a minimal CBAM implementation suitable for small backbones.
    """

    def __init__(self, channels: int, *, reduction: int = 16, spatial_kernel: int = 7) -> None:
        super().__init__()
        c = int(channels)
        r = max(1, int(reduction))
        hidden = max(8, c // r)

        self.mlp = nn.Sequential(
            nn.Conv2d(c, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, c, kernel_size=1, bias=False),
        )
        self.channel_gate = nn.Sigmoid()

        k = int(spatial_kernel)
        if k <= 0 or k % 2 == 0:
            raise ValueError("spatial_kernel must be a positive odd integer")
        self.spatial = nn.Conv2d(2, 1, kernel_size=k, padding=k // 2, bias=False)
        self.spatial_gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # --- Channel attention
        avg = torch.mean(x, dim=(2, 3), keepdim=True)
        mx = torch.amax(x, dim=(2, 3), keepdim=True)
        ca = self.channel_gate(self.mlp(avg) + self.mlp(mx))
        x = x * ca

        # --- Spatial attention
        avg = torch.mean(x, dim=1, keepdim=True)
        mx = torch.amax(x, dim=1, keepdim=True)
        sa = self.spatial_gate(self.spatial(torch.cat([avg, mx], dim=1)))
        return x * sa


class CoordAttention(nn.Module):
    """Coordinate Attention (CoordAtt).

    Pools along H and W separately, producing two attention maps.
    """

    def __init__(self, channels: int, *, reduction: int = 32) -> None:
        super().__init__()
        c = int(channels)
        r = max(1, int(reduction))
        hidden = max(8, c // r)

        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        self.conv1 = nn.Conv2d(c, hidden, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden)
        self.act = nn.Hardswish(inplace=True)

        self.conv_h = nn.Conv2d(hidden, c, kernel_size=1, bias=False)
        self.conv_w = nn.Conv2d(hidden, c, kernel_size=1, bias=False)
        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape

        x_h = self.pool_h(x)  # (B, C, H, 1)
        x_w = self.pool_w(x).transpose(2, 3)  # (B, C, W, 1)
        y = torch.cat([x_h, x_w], dim=2)  # (B, C, H+W, 1)

        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        y_h, y_w = torch.split(y, [h, w], dim=2)
        y_w = y_w.transpose(2, 3)

        a_h = self.gate(self.conv_h(y_h))
        a_w = self.gate(self.conv_w(y_w))
        return x * a_h * a_w


class GlobalContextBlock(nn.Module):
    """GCNet-style global context block (lightweight)."""

    def __init__(self, channels: int, *, reduction: int = 16) -> None:
        super().__init__()
        c = int(channels)
        r = max(1, int(reduction))
        hidden = max(8, c // r)

        self.attn = nn.Conv2d(c, 1, kernel_size=1, bias=True)
        self.transform = nn.Sequential(
            nn.Conv2d(c, hidden, kernel_size=1, bias=False),
            nn.LayerNorm([hidden, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, c, kernel_size=1, bias=False),
        )
        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        attn = self.attn(x).view(b, 1, h * w)
        attn = torch.softmax(attn, dim=-1)  # (B, 1, HW)
        context = torch.bmm(x.view(b, c, h * w), attn.transpose(1, 2)).view(b, c, 1, 1)
        y = self.transform(context)
        return x * self.gate(y)


class NonLocal2D(nn.Module):
    """Non-local block (embedded Gaussian)."""

    def __init__(self, channels: int, *, reduction: int = 2) -> None:
        super().__init__()
        c = int(channels)
        r = max(1, int(reduction))
        inter = max(8, c // r)

        self.theta = nn.Conv2d(c, inter, kernel_size=1, bias=False)
        self.phi = nn.Conv2d(c, inter, kernel_size=1, bias=False)
        self.g = nn.Conv2d(c, inter, kernel_size=1, bias=False)
        self.out = nn.Conv2d(inter, c, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        theta = self.theta(x).view(b, -1, h * w).transpose(1, 2)  # (B, HW, C')
        phi = self.phi(x).view(b, -1, h * w)  # (B, C', HW)
        attn = torch.matmul(theta, phi)  # (B, HW, HW)
        attn = torch.softmax(attn, dim=-1)

        g = self.g(x).view(b, -1, h * w).transpose(1, 2)  # (B, HW, C')
        y = torch.matmul(attn, g).transpose(1, 2).contiguous().view(b, -1, h, w)
        y = self.out(y)
        y = self.bn(y)
        return x + y


class SKConv(nn.Module):
    """Selective Kernel Convolution (SKNet-style, simplified)."""

    def __init__(self, channels: int, *, kernel_sizes: tuple[int, ...] = (3, 5), reduction: int = 16) -> None:
        super().__init__()
        c = int(channels)
        ks = tuple(int(k) for k in kernel_sizes)
        if len(ks) < 2:
            raise ValueError("kernel_sizes must contain 2+ kernels")
        r = max(1, int(reduction))
        hidden = max(8, c // r)

        self.branches = nn.ModuleList(
            [ConvBNAct(c, c, kernel_size=k, stride=1, groups=1, act="relu") for k in ks]
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Conv2d(c, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.fcs = nn.ModuleList([nn.Conv2d(hidden, c, kernel_size=1, bias=True) for _ in ks])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [b(x) for b in self.branches]
        u = torch.stack(feats, dim=0).sum(dim=0)
        s = self.pool(u)
        z = self.fc(s)
        weights = torch.stack([fc(z) for fc in self.fcs], dim=0)  # (K, B, C, 1, 1)
        weights = torch.softmax(weights, dim=0)
        y = torch.stack(feats, dim=0) * weights
        return y.sum(dim=0)


class ChannelShuffle(nn.Module):
    def __init__(self, groups: int) -> None:
        super().__init__()
        g = int(groups)
        if g <= 0:
            raise ValueError("groups must be > 0")
        self.groups = g

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        g = self.groups
        if c % g != 0:
            raise ValueError(f"channels ({c}) must be divisible by groups ({g})")
        x = x.view(b, g, c // g, h, w)
        x = x.transpose(1, 2).contiguous()
        return x.view(b, c, h, w)


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, *, stride: int = 1, act: str = "relu") -> None:
        super().__init__()
        c_in = int(in_ch)
        c_out = int(out_ch)
        self.dw = ConvBNAct(c_in, c_in, kernel_size=3, stride=int(stride), groups=c_in, act=act)
        self.pw = ConvBNAct(c_in, c_out, kernel_size=1, stride=1, padding=0, act=act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dw(x)
        return self.pw(x)


class InvertedResidual(nn.Module):
    """MobileNetV2-style inverted residual block."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        *,
        stride: int,
        expand_ratio: float = 6.0,
        se_ratio: float | None = None,
        act: str = "relu6",
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        c_in = int(in_ch)
        c_out = int(out_ch)
        s = int(stride)
        if s not in {1, 2}:
            raise ValueError("stride must be 1 or 2")

        hidden = int(round(c_in * float(expand_ratio)))
        self.use_res = s == 1 and c_in == c_out

        layers: list[nn.Module] = []
        if hidden != c_in:
            layers.append(ConvBNAct(c_in, hidden, kernel_size=1, stride=1, padding=0, act=act))
        layers.append(ConvBNAct(hidden, hidden, kernel_size=3, stride=s, groups=hidden, act=act))
        if se_ratio is not None and float(se_ratio) > 0:
            layers.append(SqueezeExcite(hidden, se_ratio=float(se_ratio)))
        # linear projection
        layers.append(nn.Conv2d(hidden, c_out, kernel_size=1, bias=False))
        layers.append(nn.BatchNorm2d(c_out))

        self.block = nn.Sequential(*layers)
        self.drop_path = DropPath(float(drop_path))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.block(x)
        if self.use_res:
            y = self.drop_path(y)
            return x + y
        return y


class GlobalAvgPoolHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, *, dropout: float = 0.0) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.drop = nn.Dropout(p=float(dropout))
        self.fc = nn.Linear(int(in_dim), int(num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        x = self.flatten(x)
        x = self.drop(x)
        return self.fc(x)


class LayerNorm2d(nn.Module):
    """LayerNorm over channels for NCHW tensors (ConvNeXt / PoolFormer style)."""

    def __init__(self, channels: int, *, eps: float = 1e-6) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(int(channels), eps=float(eps))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected NCHW tensor, got shape={tuple(x.shape)}")
        x = x.permute(0, 2, 3, 1)  # NCHW -> NHWC
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)  # NHWC -> NCHW
        return x


def init_linear_bias_from_prior(linear: nn.Linear, prior_prob: float) -> None:
    """Optional helper for detection/classification heads.

    Keeps this repo lightweight: no external initializers.
    """

    p = float(prior_prob)
    if not (0.0 < p < 1.0):
        raise ValueError("prior_prob must be in (0, 1)")
    bias = -math.log((1 - p) / p)
    nn.init.constant_(linear.bias, float(bias))
