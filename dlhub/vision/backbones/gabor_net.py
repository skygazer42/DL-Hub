import math

import torch
from torch import nn

from dlhub.vision.backbones._blocks import ConvBNAct, GlobalAvgPoolHead, scale_channels


def _gabor_kernel(
    k: int,
    *,
    sigma: float,
    theta: float,
    lambd: float,
    gamma: float,
    psi: float,
) -> torch.Tensor:
    k = int(k)
    y, x = torch.meshgrid(
        torch.arange(-(k // 2), k // 2 + 1, dtype=torch.float32),
        torch.arange(-(k // 2), k // 2 + 1, dtype=torch.float32),
        indexing="ij",
    )
    ct, st = math.cos(theta), math.sin(theta)
    x_theta = x * ct + y * st
    y_theta = -x * st + y * ct
    gb = torch.exp(-(x_theta**2 + (gamma**2) * y_theta**2) / (2 * sigma**2)) * torch.cos(
        2 * math.pi * x_theta / lambd + psi
    )
    gb = gb - gb.mean()
    gb = gb / (gb.norm() + 1e-6)
    return gb


class FixedGaborConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, *, kernel_size: int = 7, stride: int = 2) -> None:
        super().__init__()
        k = int(kernel_size)
        if k <= 0 or k % 2 == 0:
            raise ValueError("kernel_size must be positive odd")
        oc = int(out_ch)
        ic = int(in_ch)
        if oc <= 0 or ic <= 0:
            raise ValueError("channels must be > 0")

        thetas = [i * math.pi / max(1, oc) for i in range(oc)]
        kernels = []
        for i in range(oc):
            kernels.append(
                _gabor_kernel(
                    k,
                    sigma=2.0,
                    theta=thetas[i],
                    lambd=4.0,
                    gamma=0.5,
                    psi=0.0,
                )
            )
        bank = torch.stack(kernels, dim=0)  # (oc, k, k)
        bank = bank[:, None, :, :].repeat(1, ic, 1, 1)  # (oc, ic, k, k)
        self.weight = nn.Parameter(bank, requires_grad=False)
        self.stride = int(stride)
        self.padding = k // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.conv2d(
            x, self.weight, bias=None, stride=self.stride, padding=self.padding
        )


class GaborNetClassifier(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        gabor_channels: int = 32,
        channels: tuple[int, int, int] = (64, 128, 256),
        depths: tuple[int, int, int] = (2, 2, 2),
        width_mult: float = 1.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        c0 = int(gabor_channels)
        chs = tuple(scale_channels(int(c), float(width_mult)) for c in channels)
        self.stem = nn.Sequential(
            FixedGaborConv(int(in_channels), c0, kernel_size=7, stride=2),
            nn.BatchNorm2d(c0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ConvBNAct(c0, chs[0], kernel_size=3, stride=1, act="relu"),
        )

        def stage(in_ch: int, out_ch: int, depth: int, *, stride: int) -> nn.Sequential:
            layers: list[nn.Module] = [
                ConvBNAct(in_ch, out_ch, kernel_size=3, stride=int(stride), act="relu")
            ]
            for _ in range(int(depth) - 1):
                layers.append(ConvBNAct(out_ch, out_ch, kernel_size=3, stride=1, act="relu"))
            return nn.Sequential(*layers)

        self.stage1 = stage(chs[0], chs[0], depths[0], stride=1)
        self.stage2 = stage(chs[0], chs[1], depths[1], stride=2)
        self.stage3 = stage(chs[1], chs[2], depths[2], stride=2)
        self.head = GlobalAvgPoolHead(chs[2], int(num_classes), dropout=float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return self.head(x)


_VARIANTS: dict[str, dict] = {
    "gabornet_tiny": {"gabor": 24, "channels": (48, 96, 192), "depths": (2, 2, 2)},
    "gabornet_base": {"gabor": 32, "channels": (64, 128, 256), "depths": (2, 2, 2)},
    "gabornet_deep": {"gabor": 32, "channels": (64, 128, 256), "depths": (3, 3, 4)},
}


def build_gabor_net_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "gabornet_base",
    width_mult: float = 1.0,
    dropout: float = 0.1,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown GaborNet variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    return GaborNetClassifier(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        gabor_channels=int(spec["gabor"]),
        channels=tuple(map(int, spec["channels"])),
        depths=tuple(map(int, spec["depths"])),
        width_mult=float(width_mult),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_gabor_net_classifier(
        in_channels=3, num_classes=10, variant="gabornet_base", width_mult=0.5
    )
    y = m(x)
    print("gabornet_base", tuple(y.shape))
