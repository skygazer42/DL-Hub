import torch
from torch import nn

from dlhub.vision.backbones._blocks import (
    ConvBNAct,
    DepthwiseSeparableConv,
    SqueezeExcite,
    scale_channels,
)


def _c(ch: int, width_mult: float, *, min_ch: int = 8, divisor: int = 8) -> int:
    return scale_channels(int(ch), float(width_mult), min_ch=int(min_ch), divisor=int(divisor))


# ---------------------------------------------------------------------------
# MobileNetV1
# ---------------------------------------------------------------------------


class MobileNetV1Classifier(nn.Module):
    def __init__(
        self, *, in_channels: int, num_classes: int, width_mult: float, dropout: float
    ) -> None:
        super().__init__()
        w = float(width_mult)

        def c(ch: int) -> int:
            return _c(ch, w, min_ch=8, divisor=8)

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


# ---------------------------------------------------------------------------
# MobileNetV2
# ---------------------------------------------------------------------------


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
            return _c(ch, w, min_ch=8, divisor=8)

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


# ---------------------------------------------------------------------------
# MobileNetV3
# ---------------------------------------------------------------------------


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
            return _c(ch, w, min_ch=8, divisor=8)

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


# ---------------------------------------------------------------------------
# Unified builder
# ---------------------------------------------------------------------------


def build_mobilenet_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "v2",
    width_mult: float = 1.0,
    dropout: float = 0.1,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name in {"v1", "mobilenet_v1", "mobilenetv1"}:
        return build_mobilenet_v1_classifier(
            in_channels=int(in_channels),
            num_classes=int(num_classes),
            width_mult=float(width_mult),
            dropout=float(dropout),
        )
    if name in {"v2", "mobilenet_v2", "mobilenetv2"}:
        return build_mobilenet_v2_classifier(
            in_channels=int(in_channels),
            num_classes=int(num_classes),
            width_mult=float(width_mult),
            dropout=float(dropout),
        )
    if name in {"v3_small", "small"}:
        return build_mobilenet_v3_classifier(
            in_channels=int(in_channels),
            num_classes=int(num_classes),
            variant="small",
            width_mult=float(width_mult),
            dropout=float(dropout),
        )
    if name in {"v3_large", "large"}:
        return build_mobilenet_v3_classifier(
            in_channels=int(in_channels),
            num_classes=int(num_classes),
            variant="large",
            width_mult=float(width_mult),
            dropout=float(dropout),
        )
    raise ValueError("Unknown MobileNet variant. Supported: v1|v2|v3_small|v3_large")


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    for v in ["v1", "v2", "v3_small", "v3_large"]:
        m = build_mobilenet_classifier(in_channels=3, num_classes=10, variant=v, width_mult=0.75)
        y = m(x)
        print(f"mobilenet_{v}", tuple(y.shape))
