from dataclasses import dataclass

import torch
from torch import nn


def _c(ch: int, width_mult: float, *, min_ch: int = 8) -> int:
    return max(int(min_ch), int(round(int(ch) * float(width_mult))))


# ---------------------------
# VGG
# ---------------------------

_VGG_CFGS: dict[str, list[int | str]] = {
    "vgg11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "vgg13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "vgg16": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        "M",
    ],
    "vgg19": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
        "M",
    ],
}


def _make_vgg_layers(
    cfg: list[int | str], *, in_channels: int, width_mult: float
) -> tuple[nn.Sequential, int]:
    layers: list[nn.Module] = []
    c_in = int(in_channels)
    c_out_last = c_in
    for v in cfg:
        if v == "M":
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            continue
        c_out = _c(int(v), float(width_mult), min_ch=8)
        layers.extend(
            [
                nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
            ]
        )
        c_in = c_out
        c_out_last = c_out
    return nn.Sequential(*layers), int(c_out_last)


class VGGClassifier(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        cfg_name: str,
        width_mult: float,
        dropout: float,
    ) -> None:
        super().__init__()
        name = str(cfg_name).lower().strip()
        if name not in _VGG_CFGS:
            raise ValueError(f"Unknown VGG cfg: {cfg_name!r}. Supported: {sorted(_VGG_CFGS)}")

        self.features, out_ch = _make_vgg_layers(
            _VGG_CFGS[name], in_channels=int(in_channels), width_mult=float(width_mult)
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(p=float(dropout)),
            nn.Linear(out_ch, int(num_classes)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        x = self.features(x)
        return self.head(x)


def build_vgg_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str,
    width_mult: float = 1.0,
    dropout: float = 0.1,
) -> nn.Module:
    return VGGClassifier(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        cfg_name=str(variant),
        width_mult=float(width_mult),
        dropout=float(dropout),
    )


# ---------------------------
# ResNet (+ variants)
# ---------------------------


def _conv3x3(in_ch: int, out_ch: int, stride: int, *, groups: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        int(in_ch),
        int(out_ch),
        kernel_size=3,
        stride=int(stride),
        padding=1,
        groups=int(groups),
        bias=False,
    )


def _conv1x1(in_ch: int, out_ch: int, stride: int) -> nn.Conv2d:
    return nn.Conv2d(
        int(in_ch), int(out_ch), kernel_size=1, stride=int(stride), padding=0, bias=False
    )


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

    Minimal, CPU-friendly implementation:
    - Global average pool -> 1D conv over channels -> sigmoid gate.
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
        # x: (B, C, H, W)
        y = self.pool(x)  # (B, C, 1, 1)
        y = y.squeeze(-1).transpose(1, 2)  # (B, 1, C)
        y = self.conv(y)
        y = self.gate(y).transpose(1, 2).unsqueeze(-1)  # (B, C, 1, 1)
        return x * y


class _CBAMChannelAttention(nn.Module):
    def __init__(self, channels: int, *, reduction: int = 16) -> None:
        super().__init__()
        c = int(channels)
        r = max(1, int(reduction))
        hidden = max(8, c // r)

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.max = nn.AdaptiveMaxPool2d((1, 1))
        self.mlp = nn.Sequential(
            nn.Conv2d(c, hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, c, kernel_size=1, bias=True),
        )
        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.mlp(self.avg(x))
        m = self.mlp(self.max(x))
        return x * self.gate(a + m)


class _CBAMSpatialAttention(nn.Module):
    def __init__(self, *, kernel_size: int = 7) -> None:
        super().__init__()
        k = int(kernel_size)
        if k <= 0 or k % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        self.conv = nn.Conv2d(2, 1, kernel_size=k, padding=k // 2, bias=False)
        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)
        mx = x.amax(dim=1, keepdim=True)
        attn = self.conv(torch.cat([avg, mx], dim=1))
        return x * self.gate(attn)


class CBAM(nn.Module):
    """Convolutional Block Attention Module (CBAM), simplified."""

    def __init__(self, channels: int, *, reduction: int = 16, spatial_kernel: int = 7) -> None:
        super().__init__()
        self.ca = _CBAMChannelAttention(int(channels), reduction=int(reduction))
        self.sa = _CBAMSpatialAttention(kernel_size=int(spatial_kernel))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ca(x)
        x = self.sa(x)
        return x


class SKConv2d(nn.Module):
    """Selective Kernel Convolution (SKConv), simplified.

    - Two branches: 3x3 and 5x5 depth/group convs.
    - Global pooling + softmax attention to fuse branches.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        *,
        stride: int,
        groups: int = 1,
        reduction: int = 16,
    ) -> None:
        super().__init__()
        c_in = int(in_ch)
        c_out = int(out_ch)
        s = int(stride)
        g = int(groups)
        if g <= 0:
            raise ValueError("groups must be >= 1")
        if c_in % g != 0 or c_out % g != 0:
            g = 1

        hidden = max(8, c_out // max(1, int(reduction)))

        self.b1 = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=3, stride=s, padding=1, groups=g, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )
        self.b2 = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=5, stride=s, padding=2, groups=g, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Conv2d(c_out, hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
        )
        self.fc1 = nn.Conv2d(hidden, c_out, kernel_size=1, bias=True)
        self.fc2 = nn.Conv2d(hidden, c_out, kernel_size=1, bias=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u1 = self.b1(x)
        u2 = self.b2(x)
        u = u1 + u2

        s = self.fc(self.pool(u))
        a1 = self.fc1(s)
        a2 = self.fc2(s)
        a = torch.stack([a1, a2], dim=1)  # (B, 2, C, 1, 1)
        w = self.softmax(a)
        return u1 * w[:, 0] + u2 * w[:, 1]


class SplitAttentionConv2d(nn.Module):
    """Split-Attention Conv (ResNeSt-style), simplified.

    Produces `radix` splits then softmax-gates across splits.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        *,
        stride: int,
        radix: int = 2,
        groups: int = 1,
        reduction: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        c_in = int(in_ch)
        c_out = int(out_ch)
        s = int(stride)
        r = int(radix)
        g = int(groups)
        if r <= 0:
            raise ValueError("radix must be >= 1")
        if g <= 0:
            raise ValueError("groups must be >= 1")

        out_total = c_out * r
        g_try = g * r
        if c_in % g_try == 0 and out_total % g_try == 0:
            conv_groups = g_try
        elif c_in % g == 0 and out_total % g == 0:
            conv_groups = g
        else:
            conv_groups = 1

        self.radix = r
        self.out_ch = c_out
        self.conv = nn.Conv2d(
            c_in, out_total, kernel_size=3, stride=s, padding=1, groups=conv_groups, bias=False
        )
        self.bn = nn.BatchNorm2d(out_total)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout2d(p=float(dropout)) if float(dropout) > 0 else nn.Identity()

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        hidden = max(8, c_out // max(1, int(reduction)))
        self.fc1 = nn.Conv2d(c_out, hidden, kernel_size=1, bias=True)
        self.fc2 = nn.Conv2d(hidden, out_total, kernel_size=1, bias=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn(self.conv(x)))
        x = self.drop(x)
        if self.radix == 1:
            return x

        b, _, h, w = x.shape
        x = x.view(b, self.radix, self.out_ch, h, w)
        u = x.sum(dim=1)  # (B, C, H, W)

        s = self.pool(u)
        s = self.relu(self.fc1(s))
        s = self.fc2(s)  # (B, radix*C, 1, 1)
        s = s.view(b, self.radix, self.out_ch, 1, 1)
        a = self.softmax(s)
        return (x * a).sum(dim=1)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self, in_ch: int, out_ch: int, stride: int, *, groups: int = 1, width_per_group: int = 64
    ) -> None:
        super().__init__()
        _ = int(groups)
        _ = int(width_per_group)
        self.conv1 = _conv3x3(in_ch, out_ch, stride=stride, groups=1)
        self.bn1 = nn.BatchNorm2d(int(out_ch))
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = _conv3x3(out_ch, out_ch, stride=1, groups=1)
        self.bn2 = nn.BatchNorm2d(int(out_ch))

        self.downsample: nn.Module | None = None
        if int(stride) != 1 or int(in_ch) != int(out_ch):
            self.downsample = nn.Sequential(
                _conv1x1(int(in_ch), int(out_ch), stride=int(stride)),
                nn.BatchNorm2d(int(out_ch)),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(identity)

        out = out + identity
        return self.relu(out)


class ECABasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int,
        *,
        groups: int = 1,
        width_per_group: int = 64,
        eca_kernel: int = 3,
    ) -> None:
        super().__init__()
        _ = int(groups)
        _ = int(width_per_group)
        self.conv1 = _conv3x3(in_ch, out_ch, stride=stride, groups=1)
        self.bn1 = nn.BatchNorm2d(int(out_ch))
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = _conv3x3(out_ch, out_ch, stride=1, groups=1)
        self.bn2 = nn.BatchNorm2d(int(out_ch))
        self.eca = ECALayer(int(out_ch), kernel_size=int(eca_kernel))

        self.downsample: nn.Module | None = None
        if int(stride) != 1 or int(in_ch) != int(out_ch):
            self.downsample = nn.Sequential(
                _conv1x1(int(in_ch), int(out_ch), stride=int(stride)),
                nn.BatchNorm2d(int(out_ch)),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.eca(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out = out + identity
        return self.relu(out)


class CBAMBasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int,
        *,
        groups: int = 1,
        width_per_group: int = 64,
        reduction: int = 16,
    ) -> None:
        super().__init__()
        _ = int(groups)
        _ = int(width_per_group)
        self.conv1 = _conv3x3(in_ch, out_ch, stride=stride, groups=1)
        self.bn1 = nn.BatchNorm2d(int(out_ch))
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = _conv3x3(out_ch, out_ch, stride=1, groups=1)
        self.bn2 = nn.BatchNorm2d(int(out_ch))
        self.cbam = CBAM(int(out_ch), reduction=int(reduction))

        self.downsample: nn.Module | None = None
        if int(stride) != 1 or int(in_ch) != int(out_ch):
            self.downsample = nn.Sequential(
                _conv1x1(int(in_ch), int(out_ch), stride=int(stride)),
                nn.BatchNorm2d(int(out_ch)),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.cbam(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out = out + identity
        return self.relu(out)


class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int,
        *,
        groups: int = 1,
        width_per_group: int = 64,
        se_ratio: float = 0.25,
    ) -> None:
        super().__init__()
        _ = int(groups)
        _ = int(width_per_group)
        self.conv1 = _conv3x3(in_ch, out_ch, stride=stride, groups=1)
        self.bn1 = nn.BatchNorm2d(int(out_ch))
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = _conv3x3(out_ch, out_ch, stride=1, groups=1)
        self.bn2 = nn.BatchNorm2d(int(out_ch))
        self.se = SqueezeExcite(int(out_ch), se_ratio=float(se_ratio))

        self.downsample: nn.Module | None = None
        if int(stride) != 1 or int(in_ch) != int(out_ch):
            self.downsample = nn.Sequential(
                _conv1x1(int(in_ch), int(out_ch), stride=int(stride)),
                nn.BatchNorm2d(int(out_ch)),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out = out + identity
        return self.relu(out)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self, in_ch: int, out_ch: int, stride: int, *, groups: int = 1, width_per_group: int = 64
    ) -> None:
        super().__init__()
        g = int(groups)
        wpg = int(width_per_group)
        if g <= 0:
            raise ValueError("groups must be >= 1")
        if wpg <= 0:
            raise ValueError("width_per_group must be >= 1")

        # torchvision-style: width scales with groups and width_per_group.
        width = int(out_ch) * wpg // 64 * g
        width = max(g, width)

        self.conv1 = _conv1x1(in_ch, width, stride=1)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = _conv3x3(width, width, stride=stride, groups=g)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = _conv1x1(width, int(out_ch) * self.expansion, stride=1)
        self.bn3 = nn.BatchNorm2d(int(out_ch) * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        self.downsample: nn.Module | None = None
        if int(stride) != 1 or int(in_ch) != int(out_ch) * self.expansion:
            self.downsample = nn.Sequential(
                _conv1x1(int(in_ch), int(out_ch) * self.expansion, stride=int(stride)),
                nn.BatchNorm2d(int(out_ch) * self.expansion),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(identity)

        out = out + identity
        return self.relu(out)


class ECABottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int,
        *,
        groups: int = 1,
        width_per_group: int = 64,
        eca_kernel: int = 3,
    ) -> None:
        super().__init__()
        g = int(groups)
        wpg = int(width_per_group)
        if g <= 0:
            raise ValueError("groups must be >= 1")
        if wpg <= 0:
            raise ValueError("width_per_group must be >= 1")

        width = int(out_ch) * wpg // 64 * g
        width = max(g, width)

        self.conv1 = _conv1x1(in_ch, width, stride=1)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = _conv3x3(width, width, stride=stride, groups=g)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = _conv1x1(width, int(out_ch) * self.expansion, stride=1)
        self.bn3 = nn.BatchNorm2d(int(out_ch) * self.expansion)
        self.eca = ECALayer(int(out_ch) * self.expansion, kernel_size=int(eca_kernel))
        self.relu = nn.ReLU(inplace=True)

        self.downsample: nn.Module | None = None
        if int(stride) != 1 or int(in_ch) != int(out_ch) * self.expansion:
            self.downsample = nn.Sequential(
                _conv1x1(int(in_ch), int(out_ch) * self.expansion, stride=int(stride)),
                nn.BatchNorm2d(int(out_ch) * self.expansion),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = self.eca(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out = out + identity
        return self.relu(out)


class CBAMBottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int,
        *,
        groups: int = 1,
        width_per_group: int = 64,
        reduction: int = 16,
    ) -> None:
        super().__init__()
        g = int(groups)
        wpg = int(width_per_group)
        if g <= 0:
            raise ValueError("groups must be >= 1")
        if wpg <= 0:
            raise ValueError("width_per_group must be >= 1")

        width = int(out_ch) * wpg // 64 * g
        width = max(g, width)

        self.conv1 = _conv1x1(in_ch, width, stride=1)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = _conv3x3(width, width, stride=stride, groups=g)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = _conv1x1(width, int(out_ch) * self.expansion, stride=1)
        self.bn3 = nn.BatchNorm2d(int(out_ch) * self.expansion)
        self.cbam = CBAM(int(out_ch) * self.expansion, reduction=int(reduction))
        self.relu = nn.ReLU(inplace=True)

        self.downsample: nn.Module | None = None
        if int(stride) != 1 or int(in_ch) != int(out_ch) * self.expansion:
            self.downsample = nn.Sequential(
                _conv1x1(int(in_ch), int(out_ch) * self.expansion, stride=int(stride)),
                nn.BatchNorm2d(int(out_ch) * self.expansion),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = self.cbam(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out = out + identity
        return self.relu(out)


class SKBottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int,
        *,
        groups: int = 1,
        width_per_group: int = 64,
        reduction: int = 16,
    ) -> None:
        super().__init__()
        g = int(groups)
        wpg = int(width_per_group)
        if g <= 0:
            raise ValueError("groups must be >= 1")
        if wpg <= 0:
            raise ValueError("width_per_group must be >= 1")

        width = int(out_ch) * wpg // 64 * g
        width = max(g, width)

        self.conv1 = _conv1x1(in_ch, width, stride=1)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = SKConv2d(width, width, stride=int(stride), groups=g, reduction=int(reduction))
        self.conv3 = _conv1x1(width, int(out_ch) * self.expansion, stride=1)
        self.bn3 = nn.BatchNorm2d(int(out_ch) * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        self.downsample: nn.Module | None = None
        if int(stride) != 1 or int(in_ch) != int(out_ch) * self.expansion:
            self.downsample = nn.Sequential(
                _conv1x1(int(in_ch), int(out_ch) * self.expansion, stride=int(stride)),
                nn.BatchNorm2d(int(out_ch) * self.expansion),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.conv2(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(identity)

        out = out + identity
        return self.relu(out)


class ResNeStBottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int,
        *,
        groups: int = 1,
        width_per_group: int = 64,
        radix: int = 2,
        reduction: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        g = int(groups)
        wpg = int(width_per_group)
        if g <= 0:
            raise ValueError("groups must be >= 1")
        if wpg <= 0:
            raise ValueError("width_per_group must be >= 1")

        width = int(out_ch) * wpg // 64 * g
        width = max(g, width)

        self.conv1 = _conv1x1(in_ch, width, stride=1)
        self.bn1 = nn.BatchNorm2d(width)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = SplitAttentionConv2d(
            width,
            width,
            stride=int(stride),
            radix=int(radix),
            groups=g,
            reduction=int(reduction),
            dropout=float(dropout),
        )
        self.conv3 = _conv1x1(width, int(out_ch) * self.expansion, stride=1)
        self.bn3 = nn.BatchNorm2d(int(out_ch) * self.expansion)

        self.downsample: nn.Module | None = None
        if int(stride) != 1 or int(in_ch) != int(out_ch) * self.expansion:
            self.downsample = nn.Sequential(
                _conv1x1(int(in_ch), int(out_ch) * self.expansion, stride=int(stride)),
                nn.BatchNorm2d(int(out_ch) * self.expansion),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.conv2(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(identity)

        out = out + identity
        return self.relu(out)


class Res2NetBottleneck(nn.Module):
    """Res2Net bottleneck (simplified, CPU-friendly).

    Notes:
    - Keeps the external ResNet API unchanged; used as a drop-in bottleneck block.
    - For `stride=2`, uses an average-pool downsample inside the split path for shape alignment.
    """

    expansion = 4

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int,
        *,
        groups: int = 1,
        width_per_group: int = 64,
        scale: int = 4,
    ) -> None:
        super().__init__()
        g = int(groups)
        wpg = int(width_per_group)
        s = int(stride)
        if g <= 0:
            raise ValueError("groups must be >= 1")
        if wpg <= 0:
            raise ValueError("width_per_group must be >= 1")
        if s not in {1, 2}:
            raise ValueError("stride must be 1 or 2")

        sc = int(scale)
        if sc < 2:
            raise ValueError("scale must be >= 2")
        self.scale = sc

        width = int(out_ch) * wpg // 64 * g
        width = max(8, width)
        if width % sc != 0:
            width = int(((width + sc - 1) // sc) * sc)
        self.width = int(width)
        self.split_width = int(width // sc)

        self.conv1 = _conv1x1(in_ch, int(width), stride=1)
        self.bn1 = nn.BatchNorm2d(int(width))
        self.relu = nn.ReLU(inplace=True)

        # When downsampling, pool the split streams (once) so addition aligns.
        self.pool = nn.AvgPool2d(kernel_size=3, stride=s, padding=1) if s != 1 else nn.Identity()

        conv_groups = g if (self.split_width % g == 0) else 1
        self.convs = nn.ModuleList(
            [
                nn.Sequential(
                    _conv3x3(self.split_width, self.split_width, stride=1, groups=int(conv_groups)),
                    nn.BatchNorm2d(self.split_width),
                    nn.ReLU(inplace=True),
                )
                for _ in range(sc - 1)
            ]
        )

        self.conv3 = _conv1x1(int(width), int(out_ch) * self.expansion, stride=1)
        self.bn3 = nn.BatchNorm2d(int(out_ch) * self.expansion)

        self.downsample: nn.Module | None = None
        if s != 1 or int(in_ch) != int(out_ch) * self.expansion:
            self.downsample = nn.Sequential(
                _conv1x1(int(in_ch), int(out_ch) * self.expansion, stride=s),
                nn.BatchNorm2d(int(out_ch) * self.expansion),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        splits = torch.split(out, self.split_width, dim=1)
        if len(splits) != self.scale:
            raise RuntimeError("Res2Net split failed; width must be divisible by scale")
        splits = [self.pool(s) for s in splits]

        ys: list[torch.Tensor] = [splits[0]]
        prev = splits[0]
        for i in range(1, self.scale):
            cur = splits[i]
            if i > 1:
                cur = cur + prev
            cur = self.convs[i - 1](cur)
            ys.append(cur)
            prev = cur

        out = torch.cat(ys, dim=1)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(identity)

        out = out + identity
        return self.relu(out)


class SEBottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int,
        *,
        groups: int = 1,
        width_per_group: int = 64,
        se_ratio: float = 0.25,
    ) -> None:
        super().__init__()
        g = int(groups)
        wpg = int(width_per_group)
        if g <= 0:
            raise ValueError("groups must be >= 1")
        if wpg <= 0:
            raise ValueError("width_per_group must be >= 1")

        width = int(out_ch) * wpg // 64 * g
        width = max(g, width)

        self.conv1 = _conv1x1(in_ch, width, stride=1)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = _conv3x3(width, width, stride=stride, groups=g)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = _conv1x1(width, int(out_ch) * self.expansion, stride=1)
        self.bn3 = nn.BatchNorm2d(int(out_ch) * self.expansion)
        self.se = SqueezeExcite(int(out_ch) * self.expansion, se_ratio=float(se_ratio))
        self.relu = nn.ReLU(inplace=True)

        self.downsample: nn.Module | None = None
        if int(stride) != 1 or int(in_ch) != int(out_ch) * self.expansion:
            self.downsample = nn.Sequential(
                _conv1x1(int(in_ch), int(out_ch) * self.expansion, stride=int(stride)),
                nn.BatchNorm2d(int(out_ch) * self.expansion),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out = out + identity
        return self.relu(out)


class PreActBasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self, in_ch: int, out_ch: int, stride: int, *, groups: int = 1, width_per_group: int = 64
    ) -> None:
        super().__init__()
        _ = int(groups)
        _ = int(width_per_group)
        self.bn1 = nn.BatchNorm2d(int(in_ch))
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = _conv3x3(in_ch, out_ch, stride=stride, groups=1)
        self.bn2 = nn.BatchNorm2d(int(out_ch))
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = _conv3x3(out_ch, out_ch, stride=1, groups=1)

        self.downsample: nn.Module | None = None
        if int(stride) != 1 or int(in_ch) != int(out_ch):
            self.downsample = _conv1x1(int(in_ch), int(out_ch), stride=int(stride))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu1(self.bn1(x))
        identity = out if self.downsample is None else self.downsample(out)
        out = self.conv1(out)
        out = self.conv2(self.relu2(self.bn2(out)))
        out = out + identity
        return out


class PreActBottleneck(nn.Module):
    expansion = 4

    def __init__(
        self, in_ch: int, out_ch: int, stride: int, *, groups: int = 1, width_per_group: int = 64
    ) -> None:
        super().__init__()
        g = int(groups)
        wpg = int(width_per_group)
        width = int(out_ch) * wpg // 64 * g
        width = max(g, width)

        self.bn1 = nn.BatchNorm2d(int(in_ch))
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = _conv1x1(in_ch, width, stride=1)
        self.bn2 = nn.BatchNorm2d(width)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = _conv3x3(width, width, stride=stride, groups=g)
        self.bn3 = nn.BatchNorm2d(width)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv3 = _conv1x1(width, int(out_ch) * self.expansion, stride=1)

        self.downsample: nn.Module | None = None
        if int(stride) != 1 or int(in_ch) != int(out_ch) * self.expansion:
            self.downsample = _conv1x1(int(in_ch), int(out_ch) * self.expansion, stride=int(stride))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu1(self.bn1(x))
        identity = out if self.downsample is None else self.downsample(out)
        out = self.conv1(out)
        out = self.conv2(self.relu2(self.bn2(out)))
        out = self.conv3(self.relu3(self.bn3(out)))
        out = out + identity
        return out


class ResNetClassifier(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        layers: tuple[int, int, int, int],
        width_mult: float,
        dropout: float,
        block: type[nn.Module] = BasicBlock,
        groups: int = 1,
        width_per_group: int = 64,
        base_channels: int = 32,
    ) -> None:
        super().__init__()

        base = _c(int(base_channels), float(width_mult), min_ch=8)
        self.block = block
        self.groups = int(groups)
        self.width_per_group = int(width_per_group)

        self.stem = nn.Sequential(
            nn.Conv2d(int(in_channels), base, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True),
        )

        self.inplanes = base
        self.layer1 = self._make_layer(base, blocks=int(layers[0]), stride=1)
        self.layer2 = self._make_layer(base * 2, blocks=int(layers[1]), stride=2)
        self.layer3 = self._make_layer(base * 4, blocks=int(layers[2]), stride=2)
        self.layer4 = self._make_layer(base * 8, blocks=int(layers[3]), stride=2)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.drop = nn.Dropout(p=float(dropout))
        expansion = int(getattr(self.block, "expansion", 1))
        self.fc = nn.Linear(base * 8 * expansion, int(num_classes))

    def _make_layer(self, planes: int, blocks: int, stride: int) -> nn.Sequential:
        layers: list[nn.Module] = []
        layers.append(
            self.block(
                self.inplanes,
                planes,
                stride=stride,
                groups=self.groups,
                width_per_group=self.width_per_group,
            )
        )
        expansion = int(getattr(self.block, "expansion", 1))
        self.inplanes = int(planes) * expansion
        for _ in range(1, int(blocks)):
            layers.append(
                self.block(
                    self.inplanes,
                    planes,
                    stride=1,
                    groups=self.groups,
                    width_per_group=self.width_per_group,
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.drop(x)
        return self.fc(x)


def build_resnet_classifier(
    *,
    in_channels: int,
    num_classes: int,
    layers: tuple[int, int, int, int],
    variant: str,
    width_mult: float = 1.0,
    dropout: float = 0.1,
    groups: int = 1,
    width_per_group: int = 64,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name == "basic":
        block: type[nn.Module] = BasicBlock
    elif name == "bottleneck":
        block = Bottleneck
    elif name == "res2net_bottleneck":
        block = Res2NetBottleneck
    elif name == "sk_bottleneck":
        block = SKBottleneck
    elif name == "resnest_bottleneck":
        block = ResNeStBottleneck
    elif name == "eca_basic":
        block = ECABasicBlock
    elif name == "eca_bottleneck":
        block = ECABottleneck
    elif name == "cbam_basic":
        block = CBAMBasicBlock
    elif name == "cbam_bottleneck":
        block = CBAMBottleneck
    elif name == "se_basic":
        block = SEBasicBlock
    elif name == "se_bottleneck":
        block = SEBottleneck
    elif name == "preact_basic":
        block = PreActBasicBlock
    elif name == "preact_bottleneck":
        block = PreActBottleneck
    else:
        raise ValueError(
            "Unknown ResNet variant. Supported: basic, bottleneck, res2net_bottleneck, sk_bottleneck, resnest_bottleneck, eca_basic, eca_bottleneck, cbam_basic, cbam_bottleneck, se_basic, se_bottleneck, preact_basic, preact_bottleneck"
        )

    return ResNetClassifier(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        layers=(int(layers[0]), int(layers[1]), int(layers[2]), int(layers[3])),
        width_mult=float(width_mult),
        dropout=float(dropout),
        block=block,
        groups=int(groups),
        width_per_group=int(width_per_group),
        base_channels=32,
    )


# ---------------------------
# DenseNet
# ---------------------------


class DenseLayer(nn.Module):
    def __init__(self, in_ch: int, growth_rate: int, bn_size: int, dropout: float) -> None:
        super().__init__()
        inter = int(bn_size) * int(growth_rate)
        self.net = nn.Sequential(
            nn.BatchNorm2d(int(in_ch)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(in_ch), inter, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter, int(growth_rate), kernel_size=3, padding=1, bias=False),
        )
        self.drop = nn.Dropout2d(p=float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        out = self.drop(out)
        return torch.cat([x, out], dim=1)


class DenseBlock(nn.Module):
    def __init__(
        self, num_layers: int, in_ch: int, growth_rate: int, bn_size: int, dropout: float
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        c = int(in_ch)
        for _ in range(int(num_layers)):
            layers.append(
                DenseLayer(
                    c, growth_rate=int(growth_rate), bn_size=int(bn_size), dropout=float(dropout)
                )
            )
            c += int(growth_rate)
        self.layers = nn.Sequential(*layers)
        self.out_channels = c

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class Transition(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm2d(int(in_ch)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(in_ch), int(out_ch), kernel_size=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DenseNetClassifier(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        width_mult: float,
        dropout: float,
        growth_rate: int,
        block_config: tuple[int, int, int, int],
        bn_size: int = 4,
    ) -> None:
        super().__init__()

        init = _c(64, float(width_mult), min_ch=8)
        growth = max(4, int(round(int(growth_rate) * float(width_mult))))

        self.stem = nn.Sequential(
            nn.Conv2d(int(in_channels), init, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(init),
            nn.ReLU(inplace=True),
        )

        c = init
        blocks: list[nn.Module] = []
        for i, n_layers in enumerate(block_config):
            block = DenseBlock(
                num_layers=int(n_layers),
                in_ch=c,
                growth_rate=growth,
                bn_size=int(bn_size),
                dropout=float(dropout),
            )
            blocks.append(block)
            c = int(block.out_channels)
            if i != len(block_config) - 1:
                out_c = max(8, c // 2)
                blocks.append(Transition(c, out_c))
                c = out_c
        self.features = nn.Sequential(*blocks)

        self.norm = nn.BatchNorm2d(c)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(c, int(num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        x = self.stem(x)
        x = self.features(x)
        x = torch.relu(self.norm(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def build_densenet_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str,
    width_mult: float = 1.0,
    dropout: float = 0.1,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name == "densenet121":
        block_config = (6, 12, 24, 16)
        growth_rate = 32
    elif name == "densenet169":
        block_config = (6, 12, 32, 32)
        growth_rate = 32
    elif name == "densenet201":
        block_config = (6, 12, 48, 32)
        growth_rate = 32
    elif name == "densenet264":
        block_config = (6, 12, 64, 48)
        growth_rate = 32
    else:
        raise ValueError(f"Unknown DenseNet variant: {variant!r}")

    return DenseNetClassifier(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        width_mult=float(width_mult),
        dropout=float(dropout),
        growth_rate=int(growth_rate),
        block_config=tuple(map(int, block_config)),
    )


# ---------------------------
# RepVGG
# ---------------------------


def _fuse_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> tuple[torch.Tensor, torch.Tensor]:
    w = conv.weight
    if conv.bias is None:
        bias = torch.zeros(w.size(0), device=w.device, dtype=w.dtype)
    else:
        bias = conv.bias

    gamma = bn.weight
    beta = bn.bias
    mean = bn.running_mean
    var = bn.running_var
    eps = bn.eps

    std = torch.sqrt(var + eps)
    scale = (gamma / std).reshape(-1, 1, 1, 1)
    fused_w = w * scale
    fused_b = beta + (bias - mean) * (gamma / std)
    return fused_w, fused_b


def _identity_kernel(
    channels: int, kernel_size: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    k = torch.zeros((channels, channels, kernel_size, kernel_size), device=device, dtype=dtype)
    center = kernel_size // 2
    for i in range(channels):
        k[i, i, center, center] = 1.0
    return k


class RepVGGBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int, deploy: bool, dropout: float) -> None:
        super().__init__()
        self.in_ch = int(in_ch)
        self.out_ch = int(out_ch)
        self.stride = int(stride)
        self.deploy = bool(deploy)

        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout2d(p=float(dropout))

        if self.deploy:
            self.rbr_reparam = nn.Conv2d(
                self.in_ch, self.out_ch, kernel_size=3, stride=self.stride, padding=1, bias=True
            )
            self.rbr_dense = None
            self.rbr_1x1 = None
            self.rbr_identity = None
        else:
            self.rbr_reparam = None
            self.rbr_dense = nn.Sequential(
                nn.Conv2d(
                    self.in_ch,
                    self.out_ch,
                    kernel_size=3,
                    stride=self.stride,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(self.out_ch),
            )
            self.rbr_1x1 = nn.Sequential(
                nn.Conv2d(
                    self.in_ch,
                    self.out_ch,
                    kernel_size=1,
                    stride=self.stride,
                    padding=0,
                    bias=False,
                ),
                nn.BatchNorm2d(self.out_ch),
            )
            self.rbr_identity = (
                nn.BatchNorm2d(self.in_ch)
                if (self.out_ch == self.in_ch and self.stride == 1)
                else None
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        if self.deploy:
            out = self.rbr_reparam(x)
            out = self.relu(out)
            return self.drop(out)

        assert self.rbr_dense is not None and self.rbr_1x1 is not None
        out = self.rbr_dense(x) + self.rbr_1x1(x)
        if self.rbr_identity is not None:
            out = out + self.rbr_identity(x)
        out = self.relu(out)
        return self.drop(out)

    def get_equivalent_kernel_bias(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.deploy:
            conv = self.rbr_reparam
            assert conv is not None
            return conv.weight.detach().clone(), conv.bias.detach().clone()

        assert self.rbr_dense is not None and self.rbr_1x1 is not None
        k3, b3 = _fuse_conv_bn(self.rbr_dense[0], self.rbr_dense[1])
        k1, b1 = _fuse_conv_bn(self.rbr_1x1[0], self.rbr_1x1[1])
        k1 = torch.nn.functional.pad(k1, [1, 1, 1, 1])

        if self.rbr_identity is not None:
            bn = self.rbr_identity
            kid = _identity_kernel(self.in_ch, 3, device=k3.device, dtype=k3.dtype)
            gamma = bn.weight
            beta = bn.bias
            mean = bn.running_mean
            var = bn.running_var
            eps = bn.eps
            std = torch.sqrt(var + eps)
            scale = (gamma / std).reshape(-1, 1, 1, 1)
            kid = kid * scale
            bid = beta + (torch.zeros_like(mean) - mean) * (gamma / std)
        else:
            kid = torch.zeros_like(k3)
            bid = torch.zeros_like(b3)

        kernel = k3 + k1 + kid
        bias = b3 + b1 + bid
        return kernel, bias

    def switch_to_deploy(self) -> None:
        if self.deploy:
            return

        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(
            self.in_ch,
            self.out_ch,
            kernel_size=3,
            stride=self.stride,
            padding=1,
            bias=True,
        )
        self.rbr_reparam.weight.data.copy_(kernel)
        self.rbr_reparam.bias.data.copy_(bias)

        self.rbr_dense = None
        self.rbr_1x1 = None
        self.rbr_identity = None
        self.deploy = True


class RepVGGClassifier(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        width_mult: float,
        dropout: float,
        stage_blocks: tuple[int, int, int, int] = (1, 2, 4, 1),
        deploy: bool = False,
    ) -> None:
        super().__init__()
        base = _c(32, float(width_mult), min_ch=8)

        def make_stage(in_ch: int, out_ch: int, blocks: int, first_stride: int) -> nn.Sequential:
            layers: list[nn.Module] = []
            for i in range(int(blocks)):
                stride = int(first_stride) if i == 0 else 1
                layers.append(
                    RepVGGBlock(
                        in_ch=int(in_ch) if i == 0 else int(out_ch),
                        out_ch=int(out_ch),
                        stride=stride,
                        deploy=deploy,
                        dropout=float(dropout),
                    )
                )
            return nn.Sequential(*layers)

        self.stage0 = RepVGGBlock(
            in_ch=int(in_channels), out_ch=base, stride=1, deploy=deploy, dropout=float(dropout)
        )
        self.stage1 = make_stage(base, base, blocks=int(stage_blocks[0]), first_stride=1)
        self.stage2 = make_stage(base, base * 2, blocks=int(stage_blocks[1]), first_stride=2)
        self.stage3 = make_stage(base * 2, base * 4, blocks=int(stage_blocks[2]), first_stride=2)
        self.stage4 = make_stage(base * 4, base * 8, blocks=int(stage_blocks[3]), first_stride=2)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(base * 8, int(num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        x = self.stage0(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

    def switch_to_deploy(self) -> None:
        for m in self.modules():
            if isinstance(m, RepVGGBlock):
                m.switch_to_deploy()


def build_repvgg_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str,
    width_mult: float = 1.0,
    dropout: float = 0.0,
    deploy: bool = False,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name in {"repvgg_a0", "a0"}:
        stage_blocks = (1, 2, 4, 1)
    elif name in {"repvgg_a1", "a1"}:
        stage_blocks = (1, 2, 6, 2)
    elif name in {"repvgg_a2", "a2"}:
        stage_blocks = (1, 3, 8, 1)
    elif name in {"repvgg_b0", "b0"}:
        stage_blocks = (1, 4, 6, 1)
    elif name in {"repvgg_b1", "b1"}:
        stage_blocks = (2, 4, 6, 2)
    elif name in {"repvgg_b2", "b2"}:
        stage_blocks = (2, 4, 8, 2)
    elif name in {"repvgg_b3", "b3"}:
        stage_blocks = (2, 6, 10, 2)
    else:
        raise ValueError(f"Unknown RepVGG variant: {variant!r}")

    return RepVGGClassifier(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        width_mult=float(width_mult),
        dropout=float(dropout),
        stage_blocks=tuple(map(int, stage_blocks)),
        deploy=bool(deploy),
    )


# ---------------------------
# MobileNet / ShuffleNet / SqueezeNet / EfficientNet
# ---------------------------


class ConvBNAct(nn.Sequential):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        *,
        kernel_size: int,
        stride: int,
        groups: int = 1,
        act: str = "relu",
    ) -> None:
        padding = int(kernel_size) // 2
        if act == "relu6":
            act_layer: nn.Module = nn.ReLU6(inplace=True)
        elif act == "relu":
            act_layer = nn.ReLU(inplace=True)
        elif act == "hswish":
            act_layer = nn.Hardswish(inplace=True)
        else:
            raise ValueError(f"Unknown activation: {act!r}")

        super().__init__(
            nn.Conv2d(
                int(in_ch),
                int(out_ch),
                kernel_size=int(kernel_size),
                stride=int(stride),
                padding=int(padding),
                groups=int(groups),
                bias=False,
            ),
            nn.BatchNorm2d(int(out_ch)),
            act_layer,
        )


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, *, stride: int, act: str) -> None:
        super().__init__()
        self.depthwise = ConvBNAct(
            in_ch, in_ch, kernel_size=3, stride=int(stride), groups=int(in_ch), act=act
        )
        self.pointwise = ConvBNAct(in_ch, out_ch, kernel_size=1, stride=1, groups=1, act=act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        return self.pointwise(x)


class MobileNetV1Classifier(nn.Module):
    def __init__(
        self, *, in_channels: int, num_classes: int, width_mult: float, dropout: float
    ) -> None:
        super().__init__()
        w = float(width_mult)

        def c(ch: int) -> int:
            return _c(ch, w, min_ch=8)

        self.stem = ConvBNAct(int(in_channels), c(32), kernel_size=3, stride=1, act="relu")
        cfg = [
            (64, 1),
            (128, 2),
            (128, 1),
            (256, 2),
            (256, 1),
            (512, 2),
            (512, 1),
            (512, 1),
            (512, 1),
            (512, 1),
            (512, 1),
            (1024, 2),
            (1024, 1),
        ]
        layers: list[nn.Module] = []
        in_ch = c(32)
        for out_ch, stride in cfg:
            out_ch = c(out_ch)
            layers.append(DepthwiseSeparableConv(in_ch, out_ch, stride=int(stride), act="relu"))
            in_ch = out_ch
        self.features = nn.Sequential(*layers)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.drop = nn.Dropout(p=float(dropout))
        self.fc = nn.Linear(in_ch, int(num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        x = self.stem(x)
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.drop(x)
        return self.fc(x)


def build_mobilenet_v1_classifier(
    *, in_channels: int, num_classes: int, width_mult: float = 1.0, dropout: float = 0.1
) -> nn.Module:
    return MobileNetV1Classifier(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        width_mult=float(width_mult),
        dropout=float(dropout),
    )


class InvertedResidualV2(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, *, stride: int, expand_ratio: int) -> None:
        super().__init__()
        self.use_res = int(stride) == 1 and int(in_ch) == int(out_ch)

        hidden = int(in_ch) * int(expand_ratio)
        layers: list[nn.Module] = []
        if int(expand_ratio) != 1:
            layers.append(ConvBNAct(in_ch, hidden, kernel_size=1, stride=1, act="relu6"))
        layers.append(
            ConvBNAct(hidden, hidden, kernel_size=3, stride=int(stride), groups=hidden, act="relu6")
        )
        layers.append(
            nn.Sequential(
                nn.Conv2d(hidden, int(out_ch), kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(int(out_ch)),
            )
        )
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        if self.use_res:
            out = out + x
        return out


class MobileNetV2Classifier(nn.Module):
    def __init__(
        self, *, in_channels: int, num_classes: int, width_mult: float, dropout: float
    ) -> None:
        super().__init__()
        w = float(width_mult)

        def c(ch: int) -> int:
            return _c(ch, w, min_ch=8)

        self.stem = ConvBNAct(int(in_channels), c(32), kernel_size=3, stride=1, act="relu6")

        blocks_cfg = [
            # (expand_ratio, out_ch, num_blocks, stride)
            (1, 16, 1, 1),
            (6, 24, 2, 2),
            (6, 32, 3, 2),
            (6, 64, 4, 2),
            (6, 96, 3, 1),
            (6, 160, 3, 2),
            (6, 320, 1, 1),
        ]
        layers: list[nn.Module] = []
        in_ch = c(32)
        for t, out_ch, n, s in blocks_cfg:
            out_ch = c(out_ch)
            for i in range(int(n)):
                stride = int(s) if i == 0 else 1
                layers.append(InvertedResidualV2(in_ch, out_ch, stride=stride, expand_ratio=int(t)))
                in_ch = out_ch
        self.features = nn.Sequential(*layers)

        head_ch = c(1280)
        self.head = ConvBNAct(in_ch, head_ch, kernel_size=1, stride=1, act="relu6")

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.drop = nn.Dropout(p=float(dropout))
        self.fc = nn.Linear(head_ch, int(num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        x = self.stem(x)
        x = self.features(x)
        x = self.head(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.drop(x)
        return self.fc(x)


def build_mobilenet_v2_classifier(
    *, in_channels: int, num_classes: int, width_mult: float = 1.0, dropout: float = 0.1
) -> nn.Module:
    return MobileNetV2Classifier(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        width_mult=float(width_mult),
        dropout=float(dropout),
    )


class InvertedResidualV3(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        *,
        expand_ch: int,
        kernel_size: int,
        stride: int,
        se_ratio: float,
        act: str,
    ) -> None:
        super().__init__()
        self.use_res = int(stride) == 1 and int(in_ch) == int(out_ch)
        self.expand = (
            nn.Identity()
            if int(expand_ch) == int(in_ch)
            else ConvBNAct(in_ch, expand_ch, kernel_size=1, stride=1, act=act)
        )
        self.depthwise = ConvBNAct(
            int(expand_ch),
            int(expand_ch),
            kernel_size=int(kernel_size),
            stride=int(stride),
            groups=int(expand_ch),
            act=act,
        )
        self.se = (
            SqueezeExcite(int(expand_ch), se_ratio=float(se_ratio))
            if float(se_ratio) > 0
            else nn.Identity()
        )
        self.project = nn.Sequential(
            nn.Conv2d(int(expand_ch), int(out_ch), kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(int(out_ch)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.expand(x)
        out = self.depthwise(out)
        out = self.se(out)
        out = self.project(out)
        if self.use_res:
            out = out + x
        return out


class MobileNetV3Classifier(nn.Module):
    def __init__(
        self, *, in_channels: int, num_classes: int, width_mult: float, dropout: float, variant: str
    ) -> None:
        super().__init__()
        w = float(width_mult)

        def c(ch: int) -> int:
            return _c(ch, w, min_ch=8)

        name = str(variant).lower().strip()
        if name not in {"small", "large"}:
            raise ValueError("MobileNetV3 variant must be 'small' or 'large'")

        self.stem = ConvBNAct(
            int(in_channels),
            c(16),
            kernel_size=3,
            stride=1,
            act="hswish" if name == "large" else "relu",
        )

        if name == "large":
            cfg = [
                # (k, exp, out, se, act, stride)
                (3, 16, 16, 0.0, "relu", 1),
                (3, 64, 24, 0.0, "relu", 2),
                (3, 72, 24, 0.0, "relu", 1),
                (5, 72, 40, 0.25, "relu", 2),
                (5, 120, 40, 0.25, "relu", 1),
                (5, 120, 40, 0.25, "relu", 1),
                (3, 240, 80, 0.0, "hswish", 2),
                (3, 200, 80, 0.0, "hswish", 1),
                (3, 184, 80, 0.0, "hswish", 1),
                (3, 184, 80, 0.0, "hswish", 1),
                (3, 480, 112, 0.25, "hswish", 1),
                (3, 672, 112, 0.25, "hswish", 1),
                (5, 672, 160, 0.25, "hswish", 2),
                (5, 960, 160, 0.25, "hswish", 1),
                (5, 960, 160, 0.25, "hswish", 1),
            ]
            head_ch = 960
        else:
            cfg = [
                (3, 16, 16, 0.25, "relu", 2),
                (3, 72, 24, 0.0, "relu", 2),
                (3, 88, 24, 0.0, "relu", 1),
                (5, 96, 40, 0.25, "hswish", 2),
                (5, 240, 40, 0.25, "hswish", 1),
                (5, 240, 40, 0.25, "hswish", 1),
                (5, 120, 48, 0.25, "hswish", 1),
                (5, 144, 48, 0.25, "hswish", 1),
                (5, 288, 96, 0.25, "hswish", 2),
                (5, 576, 96, 0.25, "hswish", 1),
                (5, 576, 96, 0.25, "hswish", 1),
            ]
            head_ch = 576

        layers: list[nn.Module] = []
        in_ch = c(16)
        for k, exp, out, se, act, s in cfg:
            out_ch = c(out)
            exp_ch = c(exp)
            layers.append(
                InvertedResidualV3(
                    in_ch,
                    out_ch,
                    expand_ch=exp_ch,
                    kernel_size=int(k),
                    stride=int(s),
                    se_ratio=float(se),
                    act=str(act),
                )
            )
            in_ch = out_ch
        self.features = nn.Sequential(*layers)

        self.head = nn.Sequential(
            ConvBNAct(in_ch, c(head_ch), kernel_size=1, stride=1, act="hswish"),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(p=float(dropout)),
            nn.Linear(c(head_ch), int(num_classes)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        x = self.stem(x)
        x = self.features(x)
        return self.head(x)


def build_mobilenet_v3_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str,
    width_mult: float = 1.0,
    dropout: float = 0.1,
) -> nn.Module:
    return MobileNetV3Classifier(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        width_mult=float(width_mult),
        dropout=float(dropout),
        variant=str(variant),
    )


class Fire(nn.Module):
    def __init__(self, in_ch: int, squeeze: int, expand: int) -> None:
        super().__init__()
        self.squeeze = ConvBNAct(in_ch, int(squeeze), kernel_size=1, stride=1, act="relu")
        self.expand1 = ConvBNAct(int(squeeze), int(expand), kernel_size=1, stride=1, act="relu")
        self.expand3 = ConvBNAct(int(squeeze), int(expand), kernel_size=3, stride=1, act="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.squeeze(x)
        return torch.cat([self.expand1(x), self.expand3(x)], dim=1)


class SqueezeNetClassifier(nn.Module):
    def __init__(
        self, *, in_channels: int, num_classes: int, width_mult: float, dropout: float, variant: str
    ) -> None:
        super().__init__()
        w = float(width_mult)

        def c(ch: int) -> int:
            return _c(ch, w, min_ch=8)

        name = str(variant).lower().strip()
        if name not in {"1_0", "1_1"}:
            raise ValueError("SqueezeNet variant must be '1_0' or '1_1'")

        stem_out = c(32)
        self.stem = nn.Sequential(
            ConvBNAct(int(in_channels), stem_out, kernel_size=3, stride=1, act="relu"),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        layers: list[nn.Module] = []
        in_ch = stem_out
        if name == "1_0":
            stage_cfg = [
                (c(16), c(32)),
                (c(16), c(32)),
                (c(32), c(64)),
                (c(32), c(64)),
                (c(48), c(96)),
                (c(48), c(96)),
                (c(64), c(128)),
            ]
        else:
            stage_cfg = [
                (c(16), c(32)),
                (c(16), c(32)),
                (c(32), c(64)),
                (c(32), c(64)),
                (c(48), c(96)),
                (c(64), c(128)),
            ]

        for i, (sq, ex) in enumerate(stage_cfg):
            layers.append(Fire(in_ch, squeeze=sq, expand=ex))
            in_ch = 2 * int(ex)
            if i in {1, 3}:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.features = nn.Sequential(*layers)

        self.drop = nn.Dropout(p=float(dropout))
        self.classifier = nn.Conv2d(in_ch, int(num_classes), kernel_size=1, stride=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        x = self.stem(x)
        x = self.features(x)
        x = self.drop(x)
        x = self.classifier(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return x


def build_squeezenet_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str,
    width_mult: float = 1.0,
    dropout: float = 0.1,
) -> nn.Module:
    return SqueezeNetClassifier(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        width_mult=float(width_mult),
        dropout=float(dropout),
        variant=str(variant),
    )


def _channel_shuffle(x: torch.Tensor, groups: int) -> torch.Tensor:
    b, c, h, w = x.shape
    g = int(groups)
    if c % g != 0:
        raise ValueError("channels must be divisible by groups")
    x = x.view(b, g, c // g, h, w)
    x = x.transpose(1, 2).contiguous()
    return x.view(b, c, h, w)


class ShuffleV2Block(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, *, stride: int) -> None:
        super().__init__()
        s = int(stride)
        if s not in {1, 2}:
            raise ValueError("ShuffleV2Block stride must be 1 or 2")
        self.stride = s

        out_ch = int(out_ch)
        if self.stride == 1:
            if int(in_ch) != out_ch:
                raise ValueError("ShuffleV2Block stride=1 requires in_ch == out_ch")
            branch_ch = out_ch // 2
            self.branch1 = nn.Identity()
            self.branch2 = nn.Sequential(
                ConvBNAct(branch_ch, branch_ch, kernel_size=1, stride=1, act="relu"),
                ConvBNAct(
                    branch_ch, branch_ch, kernel_size=3, stride=1, groups=branch_ch, act="relu"
                ),
                ConvBNAct(branch_ch, branch_ch, kernel_size=1, stride=1, act="relu"),
            )
        else:
            branch_ch = out_ch // 2
            self.branch1 = nn.Sequential(
                ConvBNAct(in_ch, in_ch, kernel_size=3, stride=2, groups=int(in_ch), act="relu"),
                ConvBNAct(in_ch, branch_ch, kernel_size=1, stride=1, act="relu"),
            )
            self.branch2 = nn.Sequential(
                ConvBNAct(in_ch, branch_ch, kernel_size=1, stride=1, act="relu"),
                ConvBNAct(
                    branch_ch, branch_ch, kernel_size=3, stride=2, groups=branch_ch, act="relu"
                ),
                ConvBNAct(branch_ch, branch_ch, kernel_size=1, stride=1, act="relu"),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.stride == 1:
            c = x.shape[1]
            x1, x2 = x[:, : c // 2, :, :], x[:, c // 2 :, :, :]
            out = torch.cat([x1, self.branch2(x2)], dim=1)
        else:
            out = torch.cat([self.branch1(x), self.branch2(x)], dim=1)
        return _channel_shuffle(out, groups=2)


class ShuffleNetV2Classifier(nn.Module):
    def __init__(
        self, *, in_channels: int, num_classes: int, width_mult: float, dropout: float
    ) -> None:
        super().__init__()
        w = float(width_mult)
        if w <= 0.75:
            stage_out = [24, 48, 96, 192, 1024]
        elif w <= 1.25:
            stage_out = [24, 116, 232, 464, 1024]
        elif w <= 1.75:
            stage_out = [24, 176, 352, 704, 1024]
        else:
            stage_out = [24, 244, 488, 976, 2048]

        self.stem = nn.Sequential(
            ConvBNAct(int(in_channels), stage_out[0], kernel_size=3, stride=1, act="relu"),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        in_ch = stage_out[0]
        stages: list[nn.Module] = []
        for out_ch, repeats in zip(stage_out[1:4], [4, 8, 4], strict=True):
            blocks: list[nn.Module] = [ShuffleV2Block(in_ch, out_ch, stride=2)]
            for _ in range(int(repeats) - 1):
                blocks.append(ShuffleV2Block(out_ch, out_ch, stride=1))
            stages.append(nn.Sequential(*blocks))
            in_ch = out_ch
        self.stages = nn.Sequential(*stages)

        self.head = ConvBNAct(in_ch, stage_out[4], kernel_size=1, stride=1, act="relu")
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.drop = nn.Dropout(p=float(dropout))
        self.fc = nn.Linear(stage_out[4], int(num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        x = self.stem(x)
        x = self.stages(x)
        x = self.head(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.drop(x)
        return self.fc(x)


def build_shufflenet_v2_classifier(
    *, in_channels: int, num_classes: int, width_mult: float = 1.0, dropout: float = 0.1
) -> nn.Module:
    return ShuffleNetV2Classifier(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        width_mult=float(width_mult),
        dropout=float(dropout),
    )


class MBConv(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        *,
        expand_ratio: int,
        kernel_size: int,
        stride: int,
        se_ratio: float,
        act: str,
    ) -> None:
        super().__init__()
        self.use_res = int(stride) == 1 and int(in_ch) == int(out_ch)
        hidden = int(in_ch) * int(expand_ratio)

        layers: list[nn.Module] = []
        if int(expand_ratio) != 1:
            layers.append(ConvBNAct(in_ch, hidden, kernel_size=1, stride=1, act=act))
        layers.append(
            ConvBNAct(
                hidden,
                hidden,
                kernel_size=int(kernel_size),
                stride=int(stride),
                groups=hidden,
                act=act,
            )
        )
        layers.append(SqueezeExcite(hidden, se_ratio=float(se_ratio)))
        layers.append(
            nn.Sequential(
                nn.Conv2d(hidden, int(out_ch), kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(int(out_ch)),
            )
        )
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        if self.use_res:
            out = out + x
        return out


@dataclass(frozen=True)
class EfficientNetSpec:
    width_mult: float
    depth_mult: float
    dropout: float


_EFFICIENTNET_SPECS: dict[str, EfficientNetSpec] = {
    "b0": EfficientNetSpec(width_mult=1.0, depth_mult=1.0, dropout=0.2),
    "b1": EfficientNetSpec(width_mult=1.0, depth_mult=1.1, dropout=0.2),
    "b2": EfficientNetSpec(width_mult=1.1, depth_mult=1.2, dropout=0.3),
    "b3": EfficientNetSpec(width_mult=1.2, depth_mult=1.4, dropout=0.3),
    "b4": EfficientNetSpec(width_mult=1.4, depth_mult=1.8, dropout=0.4),
}


class EfficientNetClassifier(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        width_mult: float,
        depth_mult: float,
        dropout: float,
    ) -> None:
        super().__init__()
        w = float(width_mult)
        d = float(depth_mult)

        def c(ch: int) -> int:
            return _c(ch, w, min_ch=8)

        def n(repeats: int) -> int:
            return max(1, int(round(int(repeats) * d)))

        act = "relu6" if w < 0.95 else "relu"
        self.stem = ConvBNAct(int(in_channels), c(32), kernel_size=3, stride=1, act=act)

        cfg = [
            # (expand_ratio, out_ch, repeats, kernel, stride)
            (1, 16, 1, 3, 1),
            (6, 24, 2, 3, 2),
            (6, 40, 2, 5, 2),
            (6, 80, 3, 3, 2),
            (6, 112, 3, 5, 1),
            (6, 192, 4, 5, 2),
            (6, 320, 1, 3, 1),
        ]

        layers: list[nn.Module] = []
        in_ch = c(32)
        for t, out_ch, r, k, s in cfg:
            out_ch = c(out_ch)
            for i in range(n(r)):
                stride = int(s) if i == 0 else 1
                layers.append(
                    MBConv(
                        in_ch,
                        out_ch,
                        expand_ratio=int(t),
                        kernel_size=int(k),
                        stride=int(stride),
                        se_ratio=0.25,
                        act=act,
                    )
                )
                in_ch = out_ch
        self.features = nn.Sequential(*layers)

        head_ch = c(1280)
        self.head = ConvBNAct(in_ch, head_ch, kernel_size=1, stride=1, act=act)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.drop = nn.Dropout(p=float(dropout))
        self.fc = nn.Linear(head_ch, int(num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        x = self.stem(x)
        x = self.features(x)
        x = self.head(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.drop(x)
        return self.fc(x)


def build_efficientnet_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str,
    width_mult: float = 1.0,
    dropout: float | None = None,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name.startswith("efficientnet_"):
        name = name.removeprefix("efficientnet_")
    if name.startswith("b"):
        key = name
    else:
        key = name

    spec = _EFFICIENTNET_SPECS.get(key)
    if spec is None:
        raise ValueError(
            f"Unknown EfficientNet variant: {variant!r}. Supported: {sorted(_EFFICIENTNET_SPECS)}"
        )

    return EfficientNetClassifier(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        width_mult=float(width_mult) * float(spec.width_mult),
        depth_mult=float(spec.depth_mult),
        dropout=float(spec.dropout if dropout is None else dropout),
    )


# ---------------------------
# ConvNeXt
# ---------------------------


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim: int, *, layer_scale_init: float = 1e-6) -> None:
        super().__init__()
        self.dwconv = nn.Conv2d(int(dim), int(dim), kernel_size=7, padding=3, groups=int(dim))
        self.ln = nn.LayerNorm(int(dim), eps=1e-6)
        self.pw1 = nn.Linear(int(dim), 4 * int(dim))
        self.act = nn.GELU()
        self.pw2 = nn.Linear(4 * int(dim), int(dim))
        self.gamma = (
            nn.Parameter(layer_scale_init * torch.ones(int(dim))) if layer_scale_init > 0 else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # NCHW -> NHWC
        x = self.ln(x)
        x = self.pw1(x)
        x = self.act(x)
        x = self.pw2(x)
        if self.gamma is not None:
            x = x * self.gamma
        x = x.permute(0, 3, 1, 2)  # NHWC -> NCHW
        return identity + x


class LayerNorm2d(nn.Module):
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


class ConvNeXtClassifier(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        dims: tuple[int, int, int, int],
        depths: tuple[int, int, int, int],
        dropout: float,
        width_mult: float,
    ) -> None:
        super().__init__()
        dims = tuple(_c(int(d), float(width_mult), min_ch=16) for d in dims)
        depths = tuple(map(int, depths))

        self.downsample_layers = nn.ModuleList()
        self.downsample_layers.append(
            nn.Sequential(
                nn.Conv2d(int(in_channels), dims[0], kernel_size=4, stride=4),
                LayerNorm2d(dims[0], eps=1e-6),
            )
        )
        for i in range(3):
            self.downsample_layers.append(
                nn.Sequential(
                    LayerNorm2d(dims[i], eps=1e-6),
                    nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
                )
            )

        self.stages = nn.ModuleList()
        for i in range(4):
            self.stages.append(nn.Sequential(*[ConvNeXtBlock(dims[i]) for _ in range(depths[i])]))

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.drop = nn.Dropout(p=float(dropout))
        self.head = nn.Linear(dims[-1], int(num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        x = x.mean(dim=(2, 3))
        x = self.norm(x)
        x = self.drop(x)
        return self.head(x)


def build_convnext_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str,
    width_mult: float = 1.0,
    dropout: float = 0.1,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name == "convnext_tiny":
        dims = (96, 192, 384, 768)
        depths = (3, 3, 9, 3)
    elif name == "convnext_small":
        dims = (96, 192, 384, 768)
        depths = (3, 3, 27, 3)
    elif name == "convnext_base":
        dims = (128, 256, 512, 1024)
        depths = (3, 3, 27, 3)
    elif name == "convnext_large":
        dims = (192, 384, 768, 1536)
        depths = (3, 3, 27, 3)
    else:
        raise ValueError("Unknown ConvNeXt variant. Supported: convnext_tiny|small|base|large")

    return ConvNeXtClassifier(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        dims=tuple(map(int, dims)),
        depths=tuple(map(int, depths)),
        dropout=float(dropout),
        width_mult=float(width_mult),
    )
