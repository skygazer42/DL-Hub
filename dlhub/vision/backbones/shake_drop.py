import torch
from torch import nn

from dlhub.vision.backbones._blocks import ConvBNAct, GlobalAvgPoolHead, scale_channels


class ShakeDrop(nn.Module):
    """ShakeDrop regularization (simplified).

    In training, residual branch is randomly scaled (or dropped) per-sample.
    """

    def __init__(self, p_drop: float = 0.5) -> None:
        super().__init__()
        self.p_drop = float(p_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p_drop <= 0.0:
            return x
        gate = (torch.rand(x.shape[0], 1, 1, 1, device=x.device, dtype=x.dtype) > self.p_drop).to(
            x.dtype
        )
        alpha = torch.empty_like(gate).uniform_(-1.0, 1.0)
        return gate * alpha * x


class ShakeDropBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, *, stride: int, p_drop: float) -> None:
        super().__init__()
        self.res = nn.Sequential(
            ConvBNAct(int(in_ch), int(out_ch), kernel_size=3, stride=int(stride), act="relu"),
            nn.Conv2d(int(out_ch), int(out_ch), kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(int(out_ch)),
        )
        self.sd = ShakeDrop(float(p_drop))
        self.act = nn.ReLU(inplace=True)
        self.down: nn.Module | None = None
        if int(stride) != 1 or int(in_ch) != int(out_ch):
            self.down = nn.Sequential(
                nn.Conv2d(int(in_ch), int(out_ch), kernel_size=1, stride=int(stride), bias=False),
                nn.BatchNorm2d(int(out_ch)),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.down is None else self.down(x)
        y = self.sd(self.res(x))
        return self.act(identity + y)


class ShakeDropClassifier(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        channels: tuple[int, int, int, int] = (64, 128, 256, 512),
        depths: tuple[int, int, int, int] = (2, 2, 2, 2),
        p_drop: float = 0.5,
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
            blocks.append(ShakeDropBlock(in_ch, out_ch, stride=int(stride), p_drop=float(p_drop)))
            for _ in range(int(depth) - 1):
                blocks.append(ShakeDropBlock(out_ch, out_ch, stride=1, p_drop=float(p_drop)))
            return nn.Sequential(*blocks)

        self.stage1 = make_stage(chs[0], chs[0], depths[0], stride=1)
        self.stage2 = make_stage(chs[0], chs[1], depths[1], stride=2)
        self.stage3 = make_stage(chs[1], chs[2], depths[2], stride=2)
        self.stage4 = make_stage(chs[2], chs[3], depths[3], stride=2)
        self.head = GlobalAvgPoolHead(chs[3], int(num_classes), dropout=float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return self.head(x)


_VARIANTS: dict[str, dict] = {
    "shake_drop_small": {"channels": (48, 96, 192, 384), "depths": (2, 2, 2, 2), "p_drop": 0.4},
    "shake_drop_base": {"channels": (64, 128, 256, 512), "depths": (2, 2, 2, 2), "p_drop": 0.5},
    "shake_drop_deep": {"channels": (64, 128, 256, 512), "depths": (3, 4, 6, 3), "p_drop": 0.5},
}


def build_shake_drop_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "shake_drop_base",
    width_mult: float = 1.0,
    dropout: float = 0.2,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown ShakeDrop variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    return ShakeDropClassifier(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        channels=tuple(map(int, spec["channels"])),
        depths=tuple(map(int, spec["depths"])),
        p_drop=float(spec["p_drop"]),
        width_mult=float(width_mult),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_shake_drop_classifier(
        in_channels=3, num_classes=10, variant="shake_drop_base", width_mult=0.5
    )
    y = m(x)
    print("shake_drop_base", tuple(y.shape))
