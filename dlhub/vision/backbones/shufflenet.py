import torch
from torch import nn

from dlhub.vision.backbones._blocks import ConvBNAct, scale_channels


def _c(ch: int, width_mult: float, *, min_ch: int = 16, divisor: int = 8) -> int:
    return scale_channels(int(ch), float(width_mult), min_ch=int(min_ch), divisor=int(divisor))


def _channel_shuffle(x: torch.Tensor, groups: int) -> torch.Tensor:
    b, c, h, w = x.shape
    g = int(groups)
    if c % g != 0:
        raise ValueError("channels must be divisible by groups")
    x = x.view(b, g, c // g, h, w)
    x = x.transpose(1, 2).contiguous()
    return x.view(b, c, h, w)


# ---------------------------------------------------------------------------
# ShuffleNetV1 (simplified)
# ---------------------------------------------------------------------------


class ShuffleUnit(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, *, stride: int, groups: int) -> None:
        super().__init__()
        s = int(stride)
        g = int(groups)
        if s not in {1, 2}:
            raise ValueError("stride must be 1 or 2")
        mid = max(8, int(out_ch) // 4)
        g_eff = g
        if int(in_ch) % g_eff != 0 or int(mid) % g_eff != 0 or int(out_ch) % g_eff != 0:
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
        raise ValueError("Unknown ShuffleNetV1 variant. Supported: shufflenetv1_0_5|1_0|1_5|2_0")

    return ShuffleNetV1Classifier(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        width_mult=float(width_mult) * float(wm),
        dropout=float(dropout),
        groups=int(groups),
    )


# ---------------------------------------------------------------------------
# ShuffleNetV2 (simplified)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Unified builder
# ---------------------------------------------------------------------------


def build_shufflenet_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "shufflenetv2_1_0",
    width_mult: float = 1.0,
    dropout: float = 0.1,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name.startswith("shufflenetv1") or name in {"v1", "shufflenetv1"}:
        if name in {"v1", "shufflenetv1"}:
            name = "shufflenetv1_1_0"
        return build_shufflenet_v1_classifier(
            in_channels=int(in_channels),
            num_classes=int(num_classes),
            variant=name,
            width_mult=float(width_mult),
            dropout=float(dropout),
        )
    # default: v2 (use width_mult as the main scale)
    return build_shufflenet_v2_classifier(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        width_mult=float(width_mult),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m1 = build_shufflenet_classifier(
        in_channels=3, num_classes=10, variant="shufflenetv1_1_0", width_mult=1.0
    )
    m2 = build_shufflenet_classifier(
        in_channels=3, num_classes=10, variant="shufflenetv2_1_0", width_mult=1.0
    )
    print("shufflenetv1", tuple(m1(x).shape))
    print("shufflenetv2", tuple(m2(x).shape))
