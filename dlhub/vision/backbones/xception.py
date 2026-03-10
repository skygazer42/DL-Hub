import torch
from torch import nn

from dlhub.vision.backbones._blocks import ConvBNAct, scale_channels


class SeparableConvBNAct(nn.Module):
    def __init__(
        self, in_ch: int, out_ch: int, *, kernel_size: int, stride: int, act: str = "relu"
    ) -> None:
        super().__init__()
        k = int(kernel_size)
        self.depthwise = nn.Conv2d(
            int(in_ch),
            int(in_ch),
            kernel_size=k,
            stride=int(stride),
            padding=k // 2,
            groups=int(in_ch),
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(int(in_ch))
        self.pointwise = nn.Conv2d(int(in_ch), int(out_ch), kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(int(out_ch))
        act_name = str(act).lower().strip()
        self.act = nn.ReLU(inplace=True) if act_name == "relu" else nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.act(self.bn1(x))
        x = self.pointwise(x)
        x = self.act(self.bn2(x))
        return x


class XceptionBlock(nn.Module):
    def __init__(
        self, in_ch: int, out_ch: int, *, reps: int, stride: int, grow_first: bool, dropout: float
    ) -> None:
        super().__init__()
        reps = int(reps)
        stride = int(stride)
        if reps <= 0:
            raise ValueError("reps must be > 0")

        layers: list[nn.Module] = []
        c = int(in_ch)
        for i in range(reps):
            if i == 0 and grow_first:
                layers.append(
                    SeparableConvBNAct(c, int(out_ch), kernel_size=3, stride=1, act="relu")
                )
                c = int(out_ch)
            else:
                layers.append(SeparableConvBNAct(c, c, kernel_size=3, stride=1, act="relu"))
        if not grow_first:
            layers.append(SeparableConvBNAct(c, int(out_ch), kernel_size=3, stride=1, act="relu"))
            c = int(out_ch)

        self.sep = nn.Sequential(*layers)
        self.pool = (
            nn.MaxPool2d(kernel_size=3, stride=stride, padding=1) if stride != 1 else nn.Identity()
        )
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
    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        variant: str = "base",
        width_mult: float = 1.0,
        dropout: float = 0.1,
    ) -> None:
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

        stem = scale_channels(32, w, min_ch=16, divisor=8)
        self.stem = nn.Sequential(
            ConvBNAct(int(in_channels), stem, kernel_size=3, stride=2, act="relu"),
            ConvBNAct(stem, stem, kernel_size=3, stride=1, act="relu"),
        )

        blocks: list[nn.Module] = []
        in_ch = stem
        for out_base, reps in entry:
            out_ch = scale_channels(out_base, w, min_ch=32, divisor=8)
            blocks.append(
                XceptionBlock(
                    in_ch, out_ch, reps=int(reps), stride=2, grow_first=True, dropout=float(dropout)
                )
            )
            in_ch = out_ch

        for _ in range(int(middle_reps)):
            blocks.append(
                XceptionBlock(
                    in_ch, in_ch, reps=3, stride=1, grow_first=True, dropout=float(dropout)
                )
            )

        out_ch = scale_channels(int(exit_ch), w, min_ch=64, divisor=8)
        blocks.append(
            XceptionBlock(in_ch, out_ch, reps=2, stride=2, grow_first=False, dropout=float(dropout))
        )
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
    variant: str = "base",
    width_mult: float = 1.0,
    dropout: float = 0.1,
) -> nn.Module:
    return XceptionClassifier(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        variant=str(variant),
        width_mult=float(width_mult),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    for v in ["tiny", "small", "base"]:
        m = build_xception_classifier(in_channels=3, num_classes=10, variant=v, width_mult=0.5)
        y = m(x)
        print(f"xception_{v}", tuple(y.shape))
