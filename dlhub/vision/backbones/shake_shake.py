import torch
from torch import nn

from dlhub.vision.backbones._blocks import ConvBNAct, GlobalAvgPoolHead, scale_channels


class ShakeShake(nn.Module):
    """Shake-Shake combine for two residual branches (simplified).

    Original paper uses different random weights for forward/backward; here we use
    a single per-sample alpha during training and 0.5 at eval.
    """

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return 0.5 * (a + b)
        alpha = torch.rand(a.shape[0], 1, 1, 1, device=a.device, dtype=a.dtype)
        return alpha * a + (1.0 - alpha) * b


class ShakeBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, *, stride: int) -> None:
        super().__init__()
        self.branch1 = nn.Sequential(
            ConvBNAct(int(in_ch), int(out_ch), kernel_size=3, stride=int(stride), act="relu"),
            ConvBNAct(int(out_ch), int(out_ch), kernel_size=3, stride=1, act="relu"),
        )
        self.branch2 = nn.Sequential(
            ConvBNAct(int(in_ch), int(out_ch), kernel_size=3, stride=int(stride), act="relu"),
            ConvBNAct(int(out_ch), int(out_ch), kernel_size=3, stride=1, act="relu"),
        )
        self.mix = ShakeShake()
        self.act = nn.ReLU(inplace=True)
        self.down: nn.Module | None = None
        if int(stride) != 1 or int(in_ch) != int(out_ch):
            self.down = nn.Sequential(
                nn.Conv2d(int(in_ch), int(out_ch), kernel_size=1, stride=int(stride), bias=False),
                nn.BatchNorm2d(int(out_ch)),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.down is None else self.down(x)
        y = self.mix(self.branch1(x), self.branch2(x))
        return self.act(identity + y)


class ShakeShakeClassifier(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        channels: tuple[int, int, int] = (64, 128, 256),
        depths: tuple[int, int, int] = (4, 4, 4),
        width_mult: float = 1.0,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        chs = tuple(scale_channels(int(c), float(width_mult)) for c in channels)
        self.stem = nn.Sequential(
            ConvBNAct(int(in_channels), chs[0], kernel_size=3, stride=2, act="relu"),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        def make_stage(in_ch: int, out_ch: int, depth: int, *, stride: int) -> nn.Sequential:
            blocks: list[nn.Module] = []
            blocks.append(ShakeBlock(in_ch, out_ch, stride=int(stride)))
            for _ in range(int(depth) - 1):
                blocks.append(ShakeBlock(out_ch, out_ch, stride=1))
            return nn.Sequential(*blocks)

        self.stage1 = make_stage(chs[0], chs[0], depths[0], stride=1)
        self.stage2 = make_stage(chs[0], chs[1], depths[1], stride=2)
        self.stage3 = make_stage(chs[1], chs[2], depths[2], stride=2)
        self.head = GlobalAvgPoolHead(chs[2], int(num_classes), dropout=float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return self.head(x)


_VARIANTS: dict[str, dict] = {
    "shake_shake_small": {"channels": (64, 128, 256), "depths": (3, 3, 3)},
    "shake_shake_base": {"channels": (64, 128, 256), "depths": (4, 4, 4)},
    "shake_shake_large": {"channels": (96, 192, 384), "depths": (4, 4, 4)},
}


def build_shake_shake_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "shake_shake_base",
    width_mult: float = 1.0,
    dropout: float = 0.2,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown ShakeShake variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    return ShakeShakeClassifier(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        channels=tuple(map(int, spec["channels"])),
        depths=tuple(map(int, spec["depths"])),
        width_mult=float(width_mult),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_shake_shake_classifier(
        in_channels=3, num_classes=10, variant="shake_shake_base", width_mult=0.5
    )
    y = m(x)
    print("shake_shake_base", tuple(y.shape))
