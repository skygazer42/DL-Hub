
from dataclasses import dataclass

import torch
from torch import nn


def _make_divisible(v: int, divisor: int) -> int:
    d = int(divisor)
    if d <= 0:
        raise ValueError("divisor must be > 0")
    x = int(v)
    if x <= 0:
        return d
    return int((x + d - 1) // d * d)


def _c(ch: int, width_mult: float, *, min_ch: int = 8, divisor: int = 8) -> int:
    v = max(int(min_ch), int(round(int(ch) * float(width_mult))))
    return _make_divisible(v, int(divisor))


class ConvBNAct(nn.Sequential):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        *,
        kernel_size: int,
        stride: int,
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
        elif act_name in {"hswish", "hardswish"}:
            act_layer = nn.Hardswish(inplace=True)
        elif act_name in {"leaky", "leakyrelu"}:
            act_layer = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        elif act_name in {"gelu"}:
            act_layer = nn.GELU()
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


# ---------------------------------------------------------------------------
# Classic CNNs (LeNet / AlexNet / ZFNet / NiN)
# ---------------------------------------------------------------------------


class LeNet5Classifier(nn.Module):
    def __init__(self, *, in_channels: int, num_classes: int, width_mult: float, dropout: float) -> None:
        super().__init__()
        c1 = _c(16, float(width_mult), min_ch=8, divisor=4)
        c2 = _c(32, float(width_mult), min_ch=8, divisor=4)
        self.features = nn.Sequential(
            nn.Conv2d(int(in_channels), c1, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(c1, c2, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(p=float(dropout)),
            nn.Linear(c2, int(num_classes)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        x = self.features(x)
        return self.head(x)


def build_lenet_classifier(
    *, in_channels: int, num_classes: int, width_mult: float = 1.0, dropout: float = 0.1
) -> nn.Module:
    return LeNet5Classifier(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        width_mult=float(width_mult),
        dropout=float(dropout),
    )


class AlexNetClassifier(nn.Module):
    def __init__(self, *, in_channels: int, num_classes: int, width_mult: float, dropout: float) -> None:
        super().__init__()
        w = float(width_mult)
        c1 = _c(64, w, min_ch=16, divisor=8)
        c2 = _c(192, w, min_ch=32, divisor=8)
        c3 = _c(384, w, min_ch=32, divisor=8)
        c4 = _c(256, w, min_ch=32, divisor=8)
        c5 = _c(256, w, min_ch=32, divisor=8)

        self.features = nn.Sequential(
            nn.Conv2d(int(in_channels), c1, kernel_size=11, stride=2, padding=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(c1, c2, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(p=float(dropout)),
            nn.Linear(c5, int(num_classes)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        x = self.features(x)
        return self.head(x)


def build_alexnet_classifier(
    *, in_channels: int, num_classes: int, width_mult: float = 1.0, dropout: float = 0.2
) -> nn.Module:
    return AlexNetClassifier(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        width_mult=float(width_mult),
        dropout=float(dropout),
    )


class ZFNetClassifier(nn.Module):
    def __init__(self, *, in_channels: int, num_classes: int, width_mult: float, dropout: float) -> None:
        super().__init__()
        w = float(width_mult)
        c1 = _c(96, w, min_ch=16, divisor=8)
        c2 = _c(256, w, min_ch=32, divisor=8)
        c3 = _c(384, w, min_ch=32, divisor=8)
        c4 = _c(384, w, min_ch=32, divisor=8)
        c5 = _c(256, w, min_ch=32, divisor=8)

        self.features = nn.Sequential(
            nn.Conv2d(int(in_channels), c1, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(c1, c2, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(p=float(dropout)),
            nn.Linear(c5, int(num_classes)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        x = self.features(x)
        return self.head(x)


def build_zfnet_classifier(
    *, in_channels: int, num_classes: int, width_mult: float = 1.0, dropout: float = 0.2
) -> nn.Module:
    return ZFNetClassifier(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        width_mult=float(width_mult),
        dropout=float(dropout),
    )


class MLPConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, *, kernel_size: int, stride: int, padding: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            ConvBNAct(in_ch, out_ch, kernel_size=int(kernel_size), stride=int(stride), padding=int(padding), act="relu"),
            ConvBNAct(out_ch, out_ch, kernel_size=1, stride=1, padding=0, act="relu"),
            ConvBNAct(out_ch, out_ch, kernel_size=1, stride=1, padding=0, act="relu"),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class NiNClassifier(nn.Module):
    def __init__(self, *, in_channels: int, num_classes: int, width_mult: float, dropout: float) -> None:
        super().__init__()
        w = float(width_mult)
        c1 = _c(192, w, min_ch=32, divisor=8)
        c2 = _c(256, w, min_ch=32, divisor=8)
        c3 = _c(384, w, min_ch=32, divisor=8)

        self.features = nn.Sequential(
            MLPConv(int(in_channels), c1, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=float(dropout)),
            MLPConv(c1, c2, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=float(dropout)),
            MLPConv(c2, c3, kernel_size=3, stride=1, padding=1),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(c3, int(num_classes)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        x = self.features(x)
        return self.head(x)


def build_nin_classifier(
    *, in_channels: int, num_classes: int, width_mult: float = 1.0, dropout: float = 0.1
) -> nn.Module:
    return NiNClassifier(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        width_mult=float(width_mult),
        dropout=float(dropout),
    )


# ---------------------------------------------------------------------------
# Inception / GoogLeNet (simplified)
# ---------------------------------------------------------------------------


class InceptionModule(nn.Module):
    def __init__(
        self,
        in_ch: int,
        *,
        ch1x1: int,
        ch3x3_reduce: int,
        ch3x3: int,
        ch5x5_reduce: int,
        ch5x5: int,
        pool_proj: int,
    ) -> None:
        super().__init__()
        self.b1 = ConvBNAct(in_ch, ch1x1, kernel_size=1, stride=1, padding=0, act="relu")

        self.b2 = nn.Sequential(
            ConvBNAct(in_ch, ch3x3_reduce, kernel_size=1, stride=1, padding=0, act="relu"),
            ConvBNAct(ch3x3_reduce, ch3x3, kernel_size=3, stride=1, padding=1, act="relu"),
        )

        # Use two 3x3 instead of a 5x5 to keep it stable on CPU.
        self.b3 = nn.Sequential(
            ConvBNAct(in_ch, ch5x5_reduce, kernel_size=1, stride=1, padding=0, act="relu"),
            ConvBNAct(ch5x5_reduce, ch5x5, kernel_size=3, stride=1, padding=1, act="relu"),
            ConvBNAct(ch5x5, ch5x5, kernel_size=3, stride=1, padding=1, act="relu"),
        )

        self.b4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ConvBNAct(in_ch, pool_proj, kernel_size=1, stride=1, padding=0, act="relu"),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b1 = self.b1(x)
        b2 = self.b2(x)
        b3 = self.b3(x)
        b4 = self.b4(x)
        return torch.cat([b1, b2, b3, b4], dim=1)


class GoogLeNetClassifier(nn.Module):
    def __init__(self, *, in_channels: int, num_classes: int, width_mult: float, dropout: float) -> None:
        super().__init__()
        w = float(width_mult)

        stem1 = _c(64, w, min_ch=16, divisor=8)
        stem2 = _c(64, w, min_ch=16, divisor=8)
        stem3 = _c(128, w, min_ch=32, divisor=8)

        self.stem = nn.Sequential(
            ConvBNAct(int(in_channels), stem1, kernel_size=7, stride=2, padding=3, act="relu"),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ConvBNAct(stem1, stem2, kernel_size=1, stride=1, padding=0, act="relu"),
            ConvBNAct(stem2, stem3, kernel_size=3, stride=1, padding=1, act="relu"),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        c = stem3
        self.inception3a = InceptionModule(
            c,
            ch1x1=_c(64, w, min_ch=16, divisor=8),
            ch3x3_reduce=_c(96, w, min_ch=16, divisor=8),
            ch3x3=_c(128, w, min_ch=32, divisor=8),
            ch5x5_reduce=_c(16, w, min_ch=8, divisor=8),
            ch5x5=_c(32, w, min_ch=8, divisor=8),
            pool_proj=_c(32, w, min_ch=8, divisor=8),
        )
        c = (
            _c(64, w, min_ch=16, divisor=8)
            + _c(128, w, min_ch=32, divisor=8)
            + _c(32, w, min_ch=8, divisor=8)
            + _c(32, w, min_ch=8, divisor=8)
        )

        self.inception3b = InceptionModule(
            c,
            ch1x1=_c(128, w, min_ch=16, divisor=8),
            ch3x3_reduce=_c(128, w, min_ch=16, divisor=8),
            ch3x3=_c(192, w, min_ch=32, divisor=8),
            ch5x5_reduce=_c(32, w, min_ch=8, divisor=8),
            ch5x5=_c(96, w, min_ch=16, divisor=8),
            pool_proj=_c(64, w, min_ch=16, divisor=8),
        )
        c = (
            _c(128, w, min_ch=16, divisor=8)
            + _c(192, w, min_ch=32, divisor=8)
            + _c(96, w, min_ch=16, divisor=8)
            + _c(64, w, min_ch=16, divisor=8)
        )

        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception4a = InceptionModule(
            c,
            ch1x1=_c(192, w, min_ch=32, divisor=8),
            ch3x3_reduce=_c(96, w, min_ch=16, divisor=8),
            ch3x3=_c(208, w, min_ch=32, divisor=8),
            ch5x5_reduce=_c(16, w, min_ch=8, divisor=8),
            ch5x5=_c(48, w, min_ch=8, divisor=8),
            pool_proj=_c(64, w, min_ch=16, divisor=8),
        )
        c = (
            _c(192, w, min_ch=32, divisor=8)
            + _c(208, w, min_ch=32, divisor=8)
            + _c(48, w, min_ch=8, divisor=8)
            + _c(64, w, min_ch=16, divisor=8)
        )

        self.inception4b = InceptionModule(
            c,
            ch1x1=_c(160, w, min_ch=32, divisor=8),
            ch3x3_reduce=_c(112, w, min_ch=16, divisor=8),
            ch3x3=_c(224, w, min_ch=32, divisor=8),
            ch5x5_reduce=_c(24, w, min_ch=8, divisor=8),
            ch5x5=_c(64, w, min_ch=16, divisor=8),
            pool_proj=_c(64, w, min_ch=16, divisor=8),
        )
        c = (
            _c(160, w, min_ch=32, divisor=8)
            + _c(224, w, min_ch=32, divisor=8)
            + _c(64, w, min_ch=16, divisor=8)
            + _c(64, w, min_ch=16, divisor=8)
        )

        self.pool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(p=float(dropout)),
            nn.Linear(c, int(num_classes)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        x = self.stem(x)
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.pool3(x)
        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.pool4(x)
        return self.head(x)


def build_googlenet_classifier(
    *, in_channels: int, num_classes: int, width_mult: float = 1.0, dropout: float = 0.2
) -> nn.Module:
    return GoogLeNetClassifier(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        width_mult=float(width_mult),
        dropout=float(dropout),
    )


# ---------------------------------------------------------------------------
# Xception (simplified)
# ---------------------------------------------------------------------------


class SeparableConvBNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, *, kernel_size: int, stride: int, act: str) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            int(in_ch),
            int(in_ch),
            kernel_size=int(kernel_size),
            stride=int(stride),
            padding=int(kernel_size) // 2,
            groups=int(in_ch),
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(int(in_ch))
        self.pointwise = nn.Conv2d(int(in_ch), int(out_ch), kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(int(out_ch))
        self.act = nn.ReLU(inplace=True) if str(act).lower() == "relu" else nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.act(self.bn1(x))
        x = self.pointwise(x)
        x = self.act(self.bn2(x))
        return x


class XceptionBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, *, reps: int, stride: int, grow_first: bool, dropout: float) -> None:
        super().__init__()
        reps = int(reps)
        stride = int(stride)
        if reps <= 0:
            raise ValueError("reps must be > 0")

        layers: list[nn.Module] = []
        c = int(in_ch)
        for i in range(reps):
            if i == 0 and grow_first:
                layers.append(SeparableConvBNAct(c, int(out_ch), kernel_size=3, stride=1, act="relu"))
                c = int(out_ch)
            else:
                layers.append(SeparableConvBNAct(c, c, kernel_size=3, stride=1, act="relu"))
        if not grow_first:
            layers.append(SeparableConvBNAct(c, int(out_ch), kernel_size=3, stride=1, act="relu"))
            c = int(out_ch)

        self.sep = nn.Sequential(*layers)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=stride, padding=1) if stride != 1 else nn.Identity()
        self.drop = nn.Dropout2d(p=float(dropout))

        self.shortcut: nn.Module | None = None
        if stride != 1 or int(in_ch) != int(out_ch):
            self.shortcut = nn.Sequential(
                nn.Conv2d(int(in_ch), int(out_ch), kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(int(out_ch)),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.shortcut is None else self.shortcut(x)
        out = self.sep(x)
        out = self.pool(out)
        out = self.drop(out)
        return out + identity


class XceptionClassifier(nn.Module):
    def __init__(self, *, in_channels: int, num_classes: int, width_mult: float, dropout: float, variant: str) -> None:
        super().__init__()
        w = float(width_mult)
        name = str(variant).lower().strip()
        if name in {"tiny", "xception_tiny"}:
            entry = [(64, 2), (128, 2), (256, 2)]
            middle_reps = 2
            exit_ch = 512
        elif name in {"small", "xception_small"}:
            entry = [(64, 2), (128, 2), (256, 2)]
            middle_reps = 4
            exit_ch = 768
        elif name in {"base", "xception"}:
            entry = [(64, 2), (128, 2), (256, 2)]
            middle_reps = 6
            exit_ch = 1024
        else:
            raise ValueError("Unknown Xception variant. Supported: tiny|small|base")

        stem = _c(32, w, min_ch=16, divisor=8)
        self.stem = nn.Sequential(
            ConvBNAct(int(in_channels), stem, kernel_size=3, stride=2, act="relu"),
            ConvBNAct(stem, stem, kernel_size=3, stride=1, act="relu"),
        )

        blocks: list[nn.Module] = []
        in_ch = stem
        for out_base, reps in entry:
            out_ch = _c(out_base, w, min_ch=32, divisor=8)
            blocks.append(XceptionBlock(in_ch, out_ch, reps=int(reps), stride=2, grow_first=True, dropout=float(dropout)))
            in_ch = out_ch

        for _ in range(int(middle_reps)):
            blocks.append(XceptionBlock(in_ch, in_ch, reps=3, stride=1, grow_first=True, dropout=float(dropout)))

        out_ch = _c(int(exit_ch), w, min_ch=64, divisor=8)
        blocks.append(XceptionBlock(in_ch, out_ch, reps=2, stride=2, grow_first=False, dropout=float(dropout)))
        self.blocks = nn.Sequential(*blocks)

        self.head = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(p=float(dropout)),
            nn.Linear(out_ch, int(num_classes)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        x = self.stem(x)
        x = self.blocks(x)
        return self.head(x)


def build_xception_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str,
    width_mult: float = 1.0,
    dropout: float = 0.1,
) -> nn.Module:
    return XceptionClassifier(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        width_mult=float(width_mult),
        dropout=float(dropout),
        variant=str(variant),
    )


# ---------------------------------------------------------------------------
# DarkNet / CSPDarkNet (simplified)
# ---------------------------------------------------------------------------


class DarkConv(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, *, kernel_size: int, stride: int) -> None:
        super().__init__(ConvBNAct(in_ch, out_ch, kernel_size=int(kernel_size), stride=int(stride), act="leaky"))


class DarkResidual(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        c = int(channels)
        hidden = max(8, c // 2)
        self.net = nn.Sequential(
            DarkConv(c, hidden, kernel_size=1, stride=1),
            DarkConv(hidden, c, kernel_size=3, stride=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class DarkNetClassifier(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        stem_channels: int,
        stage_channels: tuple[int, int, int, int],
        stage_blocks: tuple[int, int, int, int],
        width_mult: float,
        dropout: float,
    ) -> None:
        super().__init__()
        w = float(width_mult)
        stem = _c(int(stem_channels), w, min_ch=16, divisor=8)
        self.stem = DarkConv(int(in_channels), stem, kernel_size=3, stride=1)

        in_ch = stem
        stages: list[nn.Module] = []
        for out_base, blocks in zip(stage_channels, stage_blocks, strict=True):
            out_ch = _c(int(out_base), w, min_ch=32, divisor=8)
            stage_layers: list[nn.Module] = [DarkConv(in_ch, out_ch, kernel_size=3, stride=2)]
            stage_layers.extend([DarkResidual(out_ch) for _ in range(int(blocks))])
            stages.append(nn.Sequential(*stage_layers))
            in_ch = out_ch
        self.stages = nn.Sequential(*stages)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(p=float(dropout)),
            nn.Linear(in_ch, int(num_classes)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        x = self.stem(x)
        x = self.stages(x)
        return self.head(x)


class CSPStage(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, *, num_blocks: int, width_mult: float) -> None:
        super().__init__()
        w = float(width_mult)
        out_ch = _c(int(out_ch), w, min_ch=32, divisor=8)
        hidden = max(8, out_ch // 2)

        self.down = DarkConv(int(in_ch), out_ch, kernel_size=3, stride=2)
        self.split1 = DarkConv(out_ch, hidden, kernel_size=1, stride=1)
        self.split2 = DarkConv(out_ch, hidden, kernel_size=1, stride=1)
        self.blocks = nn.Sequential(*[DarkResidual(hidden) for _ in range(int(num_blocks))])
        self.merge = DarkConv(hidden * 2, out_ch, kernel_size=1, stride=1)
        self.out_channels = out_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.down(x)
        y1 = self.blocks(self.split1(x))
        y2 = self.split2(x)
        out = torch.cat([y1, y2], dim=1)
        return self.merge(out)


class CSPDarkNetClassifier(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        stage_channels: tuple[int, int, int, int],
        stage_blocks: tuple[int, int, int, int],
        width_mult: float,
        dropout: float,
    ) -> None:
        super().__init__()
        w = float(width_mult)
        stem = _c(32, w, min_ch=16, divisor=8)
        self.stem = nn.Sequential(
            DarkConv(int(in_channels), stem, kernel_size=3, stride=1),
            DarkConv(stem, stem * 2, kernel_size=3, stride=2),
        )

        in_ch = stem * 2
        stages: list[nn.Module] = []
        for out_ch, blocks in zip(stage_channels, stage_blocks, strict=True):
            stage = CSPStage(in_ch, int(out_ch), num_blocks=int(blocks), width_mult=w)
            stages.append(stage)
            in_ch = int(stage.out_channels)
        self.stages = nn.Sequential(*stages)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(p=float(dropout)),
            nn.Linear(in_ch, int(num_classes)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        x = self.stem(x)
        x = self.stages(x)
        return self.head(x)


def build_darknet_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str,
    width_mult: float = 1.0,
    dropout: float = 0.1,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name in {"darknet19", "dn19"}:
        return DarkNetClassifier(
            in_channels=int(in_channels),
            num_classes=int(num_classes),
            stem_channels=32,
            stage_channels=(64, 128, 256, 512),
            stage_blocks=(1, 2, 2, 1),
            width_mult=float(width_mult),
            dropout=float(dropout),
        )
    if name in {"darknet53", "dn53"}:
        return DarkNetClassifier(
            in_channels=int(in_channels),
            num_classes=int(num_classes),
            stem_channels=32,
            stage_channels=(64, 128, 256, 512),
            stage_blocks=(1, 2, 8, 4),
            width_mult=float(width_mult),
            dropout=float(dropout),
        )
    if name in {"darknet_tiny", "tiny"}:
        return DarkNetClassifier(
            in_channels=int(in_channels),
            num_classes=int(num_classes),
            stem_channels=16,
            stage_channels=(32, 64, 128, 256),
            stage_blocks=(0, 1, 1, 1),
            width_mult=float(width_mult),
            dropout=float(dropout),
        )
    raise ValueError("Unknown DarkNet variant. Supported: darknet19|darknet53|darknet_tiny")


def build_cspdarknet_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str,
    width_mult: float = 1.0,
    dropout: float = 0.1,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name in {"cspdarknet53", "csp53"}:
        stage_channels = (64, 128, 256, 512)
        stage_blocks = (1, 2, 8, 4)
    elif name in {"cspdarknet_small", "small"}:
        stage_channels = (64, 128, 256, 512)
        stage_blocks = (1, 2, 4, 2)
    elif name in {"cspdarknet_tiny", "tiny"}:
        stage_channels = (32, 64, 128, 256)
        stage_blocks = (1, 1, 2, 1)
    else:
        raise ValueError("Unknown CSPDarkNet variant. Supported: cspdarknet53|cspdarknet_small|cspdarknet_tiny")

    return CSPDarkNetClassifier(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        stage_channels=tuple(map(int, stage_channels)),
        stage_blocks=tuple(map(int, stage_blocks)),
        width_mult=float(width_mult),
        dropout=float(dropout),
    )


# ---------------------------------------------------------------------------
# RegNet (config-driven, simplified)
# ---------------------------------------------------------------------------


class RegNetBottleneck(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, *, stride: int, groups: int, se_ratio: float) -> None:
        super().__init__()
        g = int(groups)
        if g <= 0:
            raise ValueError("groups must be > 0")
        self.conv1 = ConvBNAct(in_ch, out_ch, kernel_size=1, stride=1, padding=0, act="relu")
        self.conv2 = ConvBNAct(out_ch, out_ch, kernel_size=3, stride=int(stride), groups=g, act="relu")
        self.se = SqueezeExcite(out_ch, se_ratio=float(se_ratio)) if float(se_ratio) > 0 else nn.Identity()
        self.conv3 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.relu = nn.ReLU(inplace=True)

        self.down: nn.Module | None = None
        if int(stride) != 1 or int(in_ch) != int(out_ch):
            self.down = nn.Sequential(
                nn.Conv2d(int(in_ch), int(out_ch), kernel_size=1, stride=int(stride), bias=False),
                nn.BatchNorm2d(int(out_ch)),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.down is None else self.down(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.se(out)
        out = self.conv3(out)
        out = out + identity
        return self.relu(out)


class RegNetClassifier(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        widths: tuple[int, int, int, int],
        depths: tuple[int, int, int, int],
        group_width: int,
        se_ratio: float,
        width_mult: float,
        dropout: float,
    ) -> None:
        super().__init__()
        w = float(width_mult)
        widths = tuple(_c(int(x), w, min_ch=16, divisor=8) for x in widths)
        depths = tuple(map(int, depths))
        gw = max(1, int(group_width))

        self.stem = nn.Sequential(
            ConvBNAct(int(in_channels), widths[0], kernel_size=3, stride=2, act="relu"),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        in_ch = widths[0]
        stages: list[nn.Module] = []
        for i, (out_ch, d) in enumerate(zip(widths, depths, strict=True)):
            stride = 1 if i == 0 else 2
            groups = max(1, int(out_ch) // gw)
            if int(out_ch) % groups != 0:
                groups = 1

            blocks: list[nn.Module] = [
                RegNetBottleneck(in_ch, int(out_ch), stride=stride, groups=groups, se_ratio=float(se_ratio))
            ]
            for _ in range(1, int(d)):
                blocks.append(RegNetBottleneck(int(out_ch), int(out_ch), stride=1, groups=groups, se_ratio=float(se_ratio)))
            stages.append(nn.Sequential(*blocks))
            in_ch = int(out_ch)
        self.stages = nn.Sequential(*stages)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(p=float(dropout)),
            nn.Linear(in_ch, int(num_classes)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        x = self.stem(x)
        x = self.stages(x)
        return self.head(x)


@dataclass(frozen=True)
class _RegNetSizeSpec:
    width_mult: float
    depths: tuple[int, int, int, int]


_REGNET_SIZES: dict[str, _RegNetSizeSpec] = {
    "200mf": _RegNetSizeSpec(width_mult=0.65, depths=(1, 1, 4, 1)),
    "400mf": _RegNetSizeSpec(width_mult=0.8, depths=(1, 2, 4, 1)),
    "600mf": _RegNetSizeSpec(width_mult=0.9, depths=(2, 2, 5, 1)),
    "800mf": _RegNetSizeSpec(width_mult=1.0, depths=(2, 2, 6, 2)),
    "1_6gf": _RegNetSizeSpec(width_mult=1.2, depths=(2, 3, 8, 2)),
    "2_4gf": _RegNetSizeSpec(width_mult=1.3, depths=(3, 3, 9, 2)),
    "3_2gf": _RegNetSizeSpec(width_mult=1.4, depths=(3, 4, 10, 2)),
    "4gf": _RegNetSizeSpec(width_mult=1.55, depths=(3, 4, 12, 3)),
    "6_4gf": _RegNetSizeSpec(width_mult=1.75, depths=(3, 5, 14, 3)),
    "8gf": _RegNetSizeSpec(width_mult=1.95, depths=(3, 6, 16, 3)),
    "12gf": _RegNetSizeSpec(width_mult=2.25, depths=(4, 7, 20, 4)),
    "16gf": _RegNetSizeSpec(width_mult=2.55, depths=(4, 8, 24, 4)),
    "32gf": _RegNetSizeSpec(width_mult=3.1, depths=(5, 10, 30, 5)),
}


def build_regnet_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str,
    width_mult: float = 1.0,
    dropout: float = 0.1,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name.startswith("regnet"):
        name = name.removeprefix("regnet")
    name = name.lstrip("_")

    if name.startswith("x_"):
        family = "x"
        key = name.removeprefix("x_")
        se_ratio = 0.0
        group_width = 16
    elif name.startswith("y_"):
        family = "y"
        key = name.removeprefix("y_")
        se_ratio = 0.25
        group_width = 8
    else:
        raise ValueError("RegNet variant must start with x_ or y_. Example: regnetx_400mf or regnety_3_2gf")

    spec = _REGNET_SIZES.get(key)
    if spec is None:
        raise ValueError(f"Unknown RegNet{family} size: {key!r}. Supported: {sorted(_REGNET_SIZES)}")

    base_widths = (64, 128, 256, 512)
    widths = tuple(_c(int(wi), float(spec.width_mult), min_ch=16, divisor=8) for wi in base_widths)

    return RegNetClassifier(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        widths=tuple(map(int, widths)),
        depths=tuple(map(int, spec.depths)),
        group_width=int(group_width),
        se_ratio=float(se_ratio),
        width_mult=float(width_mult),
        dropout=float(dropout),
    )


# ---------------------------------------------------------------------------
# GhostNet (simplified)
# ---------------------------------------------------------------------------


def _cheap_depthwise(in_ch: int, out_ch: int, *, kernel_size: int = 3, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(int(in_ch), int(out_ch), kernel_size=int(kernel_size), stride=int(stride), padding=int(kernel_size) // 2, groups=int(in_ch), bias=False),
        nn.BatchNorm2d(int(out_ch)),
        nn.ReLU(inplace=True),
    )


class GhostModule(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, *, ratio: int = 2) -> None:
        super().__init__()
        out_ch = int(out_ch)
        primary = int((out_ch + int(ratio) - 1) // int(ratio))
        cheap = int(out_ch - primary)
        self.primary = nn.Sequential(
            nn.Conv2d(int(in_ch), primary, kernel_size=1, bias=False),
            nn.BatchNorm2d(primary),
            nn.ReLU(inplace=True),
        )
        self.cheap = _cheap_depthwise(primary, cheap, kernel_size=3, stride=1) if cheap > 0 else nn.Identity()
        self.out_ch = out_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.primary(x)
        x2 = self.cheap(x1)
        if isinstance(x2, torch.Tensor):
            out = torch.cat([x1, x2], dim=1)
        else:
            out = x1
        return out[:, : self.out_ch, :, :]


class GhostBottleneck(nn.Module):
    def __init__(self, in_ch: int, mid_ch: int, out_ch: int, *, stride: int, se_ratio: float) -> None:
        super().__init__()
        self.stride = int(stride)
        self.ghost1 = GhostModule(in_ch, mid_ch)
        self.dw = (
            nn.Sequential(
                nn.Conv2d(int(mid_ch), int(mid_ch), kernel_size=3, stride=int(stride), padding=1, groups=int(mid_ch), bias=False),
                nn.BatchNorm2d(int(mid_ch)),
            )
            if int(stride) != 1
            else nn.Identity()
        )
        self.se = SqueezeExcite(int(mid_ch), se_ratio=float(se_ratio)) if float(se_ratio) > 0 else nn.Identity()
        self.ghost2 = GhostModule(mid_ch, out_ch)

        self.shortcut: nn.Module
        if int(stride) == 1 and int(in_ch) == int(out_ch):
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(int(in_ch), int(in_ch), kernel_size=3, stride=int(stride), padding=1, groups=int(in_ch), bias=False),
                nn.BatchNorm2d(int(in_ch)),
                nn.Conv2d(int(in_ch), int(out_ch), kernel_size=1, bias=False),
                nn.BatchNorm2d(int(out_ch)),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.ghost1(x)
        out = self.dw(out)
        out = self.se(out)
        out = self.ghost2(out)
        return out + self.shortcut(x)


class GhostNetClassifier(nn.Module):
    def __init__(self, *, in_channels: int, num_classes: int, width_mult: float, dropout: float, se_ratio: float) -> None:
        super().__init__()
        w = float(width_mult)

        def c(ch: int) -> int:
            return _c(ch, w, min_ch=8, divisor=8)

        self.stem = nn.Sequential(
            ConvBNAct(int(in_channels), c(16), kernel_size=3, stride=2, act="relu"),
        )

        cfg = [
            # (k, exp, out, se, s)
            (3, 16, 16, 0.0, 1),
            (3, 48, 24, float(se_ratio), 2),
            (3, 72, 24, 0.0, 1),
            (5, 72, 40, float(se_ratio), 2),
            (5, 120, 40, float(se_ratio), 1),
            (3, 240, 80, 0.0, 2),
            (3, 200, 80, 0.0, 1),
            (3, 184, 80, 0.0, 1),
            (3, 184, 80, 0.0, 1),
            (3, 480, 112, float(se_ratio), 1),
            (5, 672, 160, float(se_ratio), 2),
            (5, 960, 160, float(se_ratio), 1),
        ]

        layers: list[nn.Module] = []
        in_ch = c(16)
        for _k, exp, out, se, s in cfg:
            layers.append(GhostBottleneck(in_ch, c(exp), c(out), stride=int(s), se_ratio=float(se)))
            in_ch = c(out)
        self.features = nn.Sequential(*layers)

        head_ch = c(960)
        self.head = nn.Sequential(
            ConvBNAct(in_ch, head_ch, kernel_size=1, stride=1, padding=0, act="relu"),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(p=float(dropout)),
            nn.Linear(head_ch, int(num_classes)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        x = self.stem(x)
        x = self.features(x)
        return self.head(x)


def build_ghostnet_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str,
    width_mult: float = 1.0,
    dropout: float = 0.1,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name in {"ghostnet", "ghostnet_1_0", "1_0"}:
        wm = 1.0
        se_ratio = 0.25
    elif name in {"ghostnet_0_5", "0_5"}:
        wm = 0.5
        se_ratio = 0.25
    elif name in {"ghostnet_0_75", "0_75"}:
        wm = 0.75
        se_ratio = 0.25
    elif name in {"ghostnet_1_3", "1_3"}:
        wm = 1.3
        se_ratio = 0.25
    elif name in {"ghostnet_1_5", "1_5"}:
        wm = 1.5
        se_ratio = 0.25
    elif name in {"ghostnetv2_1_0", "v2_1_0"}:
        wm = 1.0
        se_ratio = 0.25
    elif name in {"ghostnetv2_1_3", "v2_1_3"}:
        wm = 1.3
        se_ratio = 0.25
    else:
        raise ValueError(
            "Unknown GhostNet variant. Supported: ghostnet_0_5|0_75|1_0|1_3|1_5|ghostnetv2_1_0|ghostnetv2_1_3"
        )

    return GhostNetClassifier(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        width_mult=float(width_mult) * float(wm),
        dropout=float(dropout),
        se_ratio=float(se_ratio),
    )


# ---------------------------------------------------------------------------
# ShuffleNetV1 (simplified)
# ---------------------------------------------------------------------------


def _channel_shuffle(x: torch.Tensor, groups: int) -> torch.Tensor:
    b, c, h, w = x.shape
    g = int(groups)
    if c % g != 0:
        raise ValueError("channels must be divisible by groups")
    x = x.view(b, g, c // g, h, w)
    x = x.transpose(1, 2).contiguous()
    return x.view(b, c, h, w)


class ShuffleUnit(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, *, stride: int, groups: int) -> None:
        super().__init__()
        s = int(stride)
        g = int(groups)
        if s not in {1, 2}:
            raise ValueError("stride must be 1 or 2")
        mid = max(8, int(out_ch) // 4)
        g_eff = g
        if (
            int(in_ch) % g_eff != 0
            or int(mid) % g_eff != 0
            or int(out_ch) % g_eff != 0
        ):
            g_eff = 1

        self.stride = s
        self.groups = g_eff
        self.gconv1 = nn.Sequential(
            nn.Conv2d(int(in_ch), mid, kernel_size=1, groups=g_eff, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
        )
        self.dwconv = nn.Sequential(
            nn.Conv2d(mid, mid, kernel_size=3, stride=s, padding=1, groups=mid, bias=False),
            nn.BatchNorm2d(mid),
        )
        self.gconv2 = nn.Sequential(
            nn.Conv2d(mid, int(out_ch), kernel_size=1, groups=g_eff, bias=False),
            nn.BatchNorm2d(int(out_ch)),
        )

        self.proj = (
            nn.Sequential(
                nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
                nn.Conv2d(int(in_ch), int(out_ch), kernel_size=1, bias=False),
                nn.BatchNorm2d(int(out_ch)),
            )
            if s == 2
            else nn.Identity()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.gconv1(x)
        out = _channel_shuffle(out, groups=self.groups)
        out = self.dwconv(out)
        out = self.gconv2(out)
        out = out + self.proj(x)
        return self.relu(out)


class ShuffleNetV1Classifier(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        width_mult: float,
        dropout: float,
        groups: int,
    ) -> None:
        super().__init__()
        w = float(width_mult)
        g = int(groups)

        def c(ch: int) -> int:
            return _c(ch, w, min_ch=16, divisor=8)

        self.stem = nn.Sequential(
            ConvBNAct(int(in_channels), c(24), kernel_size=3, stride=2, act="relu"),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        stage_out = [c(240), c(480), c(960)]
        repeats = [4, 8, 4]

        in_ch = c(24)
        stages: list[nn.Module] = []
        for out_ch, r in zip(stage_out, repeats, strict=True):
            blocks: list[nn.Module] = [ShuffleUnit(in_ch, out_ch, stride=2, groups=g)]
            for _ in range(int(r) - 1):
                blocks.append(ShuffleUnit(out_ch, out_ch, stride=1, groups=g))
            stages.append(nn.Sequential(*blocks))
            in_ch = out_ch
        self.stages = nn.Sequential(*stages)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(p=float(dropout)),
            nn.Linear(in_ch, int(num_classes)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        x = self.stem(x)
        x = self.stages(x)
        return self.head(x)


def build_shufflenet_v1_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str,
    width_mult: float = 1.0,
    dropout: float = 0.1,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name in {"shufflenetv1_0_5", "0_5"}:
        wm = 0.5
        groups = 2
    elif name in {"shufflenetv1_1_0", "1_0", "shufflenetv1"}:
        wm = 1.0
        groups = 3
    elif name in {"shufflenetv1_1_5", "1_5"}:
        wm = 1.5
        groups = 3
    elif name in {"shufflenetv1_2_0", "2_0"}:
        wm = 2.0
        groups = 4
    else:
        raise ValueError("Unknown ShuffleNetV1 variant. Supported: 0_5|1_0|1_5|2_0")

    return ShuffleNetV1Classifier(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        width_mult=float(width_mult) * float(wm),
        dropout=float(dropout),
        groups=int(groups),
    )


# ---------------------------------------------------------------------------
# MNASNet (simplified)
# ---------------------------------------------------------------------------


class InvertedResidual(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, *, stride: int, expand_ratio: int, kernel_size: int, se_ratio: float) -> None:
        super().__init__()
        self.use_res = int(stride) == 1 and int(in_ch) == int(out_ch)
        hidden = int(in_ch) * int(expand_ratio)

        layers: list[nn.Module] = []
        if hidden != int(in_ch):
            layers.append(ConvBNAct(in_ch, hidden, kernel_size=1, stride=1, padding=0, act="relu"))
        layers.append(ConvBNAct(hidden, hidden, kernel_size=int(kernel_size), stride=int(stride), groups=hidden, act="relu"))
        if float(se_ratio) > 0:
            layers.append(SqueezeExcite(hidden, se_ratio=float(se_ratio)))
        layers.append(
            nn.Sequential(
                nn.Conv2d(hidden, int(out_ch), kernel_size=1, bias=False),
                nn.BatchNorm2d(int(out_ch)),
            )
        )
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        if self.use_res:
            out = out + x
        return out


class MNASNetClassifier(nn.Module):
    def __init__(self, *, in_channels: int, num_classes: int, width_mult: float, dropout: float, variant: str) -> None:
        super().__init__()
        w = float(width_mult)
        name = str(variant).lower().strip()
        if name in {"0_5", "mnasnet0_5"}:
            wm = 0.5
        elif name in {"0_75", "mnasnet0_75"}:
            wm = 0.75
        elif name in {"1_0", "mnasnet1_0", "mnasnet"}:
            wm = 1.0
        elif name in {"1_3", "mnasnet1_3"}:
            wm = 1.3
        else:
            raise ValueError("Unknown MNASNet variant. Supported: 0_5|0_75|1_0|1_3")

        def c(ch: int) -> int:
            return _c(ch, w * wm, min_ch=8, divisor=8)

        self.stem = ConvBNAct(int(in_channels), c(32), kernel_size=3, stride=2, act="relu")

        cfg = [
            # (k, exp, out, n, s, se)
            (3, 1, 16, 1, 1, 0.0),
            (3, 6, 24, 3, 2, 0.0),
            (5, 3, 40, 3, 2, 0.25),
            (3, 6, 80, 3, 2, 0.0),
            (5, 6, 112, 2, 1, 0.25),
            (5, 6, 160, 4, 2, 0.25),
            (3, 6, 320, 1, 1, 0.0),
        ]

        layers: list[nn.Module] = []
        in_ch = c(32)
        for k, t, out, n, s, se in cfg:
            out_ch = c(out)
            for i in range(int(n)):
                stride = int(s) if i == 0 else 1
                layers.append(
                    InvertedResidual(in_ch, out_ch, stride=stride, expand_ratio=int(t), kernel_size=int(k), se_ratio=float(se))
                )
                in_ch = out_ch
        self.features = nn.Sequential(*layers)

        head_ch = c(1280)
        self.head = nn.Sequential(
            ConvBNAct(in_ch, head_ch, kernel_size=1, stride=1, padding=0, act="relu"),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(p=float(dropout)),
            nn.Linear(head_ch, int(num_classes)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        x = self.stem(x)
        x = self.features(x)
        return self.head(x)


def build_mnasnet_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str,
    width_mult: float = 1.0,
    dropout: float = 0.1,
) -> nn.Module:
    return MNASNetClassifier(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        width_mult=float(width_mult),
        dropout=float(dropout),
        variant=str(variant),
    )


# ---------------------------------------------------------------------------
# MobileOne (very small, RepVGG-like)
# ---------------------------------------------------------------------------


class MobileOneBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, *, stride: int, num_branches: int, dropout: float) -> None:
        super().__init__()
        self.in_ch = int(in_ch)
        self.out_ch = int(out_ch)
        self.stride = int(stride)

        branches: list[nn.Module] = []
        for _ in range(int(num_branches)):
            branches.append(
                nn.Sequential(
                    nn.Conv2d(self.in_ch, self.out_ch, kernel_size=3, stride=self.stride, padding=1, bias=False),
                    nn.BatchNorm2d(self.out_ch),
                )
            )
        self.branches = nn.ModuleList(branches)
        self.branch_1x1 = nn.Sequential(
            nn.Conv2d(self.in_ch, self.out_ch, kernel_size=1, stride=self.stride, padding=0, bias=False),
            nn.BatchNorm2d(self.out_ch),
        )
        self.identity = (
            nn.BatchNorm2d(self.in_ch) if (self.in_ch == self.out_ch and self.stride == 1) else None
        )
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout2d(p=float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        out = self.branch_1x1(x)
        for br in self.branches:
            out = out + br(x)
        if self.identity is not None:
            out = out + self.identity(x)
        out = self.relu(out)
        return self.drop(out)


class MobileOneClassifier(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        width_mult: float,
        dropout: float,
        stage_blocks: tuple[int, int, int, int],
        num_branches: int,
        use_se: bool,
    ) -> None:
        super().__init__()
        w = float(width_mult)

        def c(ch: int) -> int:
            return _c(ch, w, min_ch=8, divisor=8)

        base = c(32)
        self.stem = MobileOneBlock(int(in_channels), base, stride=2, num_branches=int(num_branches), dropout=float(dropout))

        def make_stage(in_ch: int, out_ch: int, blocks: int, first_stride: int) -> nn.Sequential:
            layers: list[nn.Module] = []
            for i in range(int(blocks)):
                stride = int(first_stride) if i == 0 else 1
                layers.append(MobileOneBlock(int(in_ch) if i == 0 else int(out_ch), int(out_ch), stride=stride, num_branches=int(num_branches), dropout=float(dropout)))
                if use_se:
                    layers.append(SqueezeExcite(int(out_ch), se_ratio=0.25))
            return nn.Sequential(*layers)

        self.stage1 = make_stage(base, c(64), blocks=int(stage_blocks[0]), first_stride=1)
        self.stage2 = make_stage(c(64), c(128), blocks=int(stage_blocks[1]), first_stride=2)
        self.stage3 = make_stage(c(128), c(256), blocks=int(stage_blocks[2]), first_stride=2)
        self.stage4 = make_stage(c(256), c(512), blocks=int(stage_blocks[3]), first_stride=2)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(p=float(dropout)),
            nn.Linear(c(512), int(num_classes)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return self.head(x)


def build_mobileone_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str,
    width_mult: float = 1.0,
    dropout: float = 0.1,
) -> nn.Module:
    name = str(variant).lower().strip()
    # Very small "s" variants: depth scales; all use the same block type.
    if name in {"s0", "mobileone_s0"}:
        stage_blocks = (1, 1, 3, 1)
        branches = 1
    elif name in {"s1", "mobileone_s1"}:
        stage_blocks = (1, 2, 4, 1)
        branches = 1
    elif name in {"s2", "mobileone_s2"}:
        stage_blocks = (1, 2, 6, 2)
        branches = 2
    elif name in {"s3", "mobileone_s3"}:
        stage_blocks = (2, 3, 8, 2)
        branches = 2
    elif name in {"s4", "mobileone_s4"}:
        stage_blocks = (2, 4, 10, 2)
        branches = 3
    elif name in {"s1_se", "mobileone_s1_se"}:
        stage_blocks = (1, 2, 4, 1)
        branches = 1
    elif name in {"s2_se", "mobileone_s2_se"}:
        stage_blocks = (1, 2, 6, 2)
        branches = 2
    elif name in {"s3_se", "mobileone_s3_se"}:
        stage_blocks = (2, 3, 8, 2)
        branches = 2
    elif name in {"s4_se", "mobileone_s4_se"}:
        stage_blocks = (2, 4, 10, 2)
        branches = 3
    else:
        raise ValueError("Unknown MobileOne variant. Supported: s0|s1|s2|s3|s4 (+ _se variants)")

    use_se = name.endswith("_se")

    return MobileOneClassifier(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        width_mult=float(width_mult),
        dropout=float(dropout),
        stage_blocks=tuple(map(int, stage_blocks)),
        num_branches=int(branches),
        use_se=bool(use_se),
    )


__all__ = [
    "build_alexnet_classifier",
    "build_cspdarknet_classifier",
    "build_darknet_classifier",
    "build_ghostnet_classifier",
    "build_googlenet_classifier",
    "build_lenet_classifier",
    "build_mnasnet_classifier",
    "build_mobileone_classifier",
    "build_nin_classifier",
    "build_regnet_classifier",
    "build_shufflenet_v1_classifier",
    "build_xception_classifier",
    "build_zfnet_classifier",
]
