import torch
import torch.nn.functional as F
from torch import nn

from dlhub.vision.backbones._blocks import ConvBNAct
from dlhub.vision.segmentation._common import check_nchw


class _EncStage(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, *, depth: int) -> None:
        super().__init__()
        c_in = int(in_ch)
        c_out = int(out_ch)
        d = int(depth)
        if d <= 0:
            raise ValueError("depth must be > 0")
        layers: list[nn.Module] = [ConvBNAct(c_in, c_out, kernel_size=3, stride=2, act="relu")]
        for _ in range(d - 1):
            layers.append(ConvBNAct(c_out, c_out, kernel_size=3, stride=1, act="relu"))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _DecStage(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, *, depth: int) -> None:
        super().__init__()
        c_in = int(in_ch)
        c_out = int(out_ch)
        d = int(depth)
        if d <= 0:
            raise ValueError("depth must be > 0")
        self.up = nn.ConvTranspose2d(c_in, c_out, kernel_size=2, stride=2, bias=True)
        layers: list[nn.Module] = [ConvBNAct(c_out, c_out, kernel_size=3, stride=1, act="relu")]
        for _ in range(d - 1):
            layers.append(ConvBNAct(c_out, c_out, kernel_size=3, stride=1, act="relu"))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
        x = x + skip
        return self.net(x)


class LinkNet(nn.Module):
    """LinkNet semantic segmentation (toy-first, pure torch).

    Uses additive skip connections (vs U-Net concatenation).
    """

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        base_channels: int = 32,
        levels: int = 4,
        depth: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        nc = int(num_classes)
        if nc <= 0:
            raise ValueError("num_classes must be > 0")
        base = int(base_channels)
        lv = int(levels)
        d = int(depth)
        if lv < 2:
            raise ValueError("levels must be >= 2")

        self.stem = nn.Sequential(
            ConvBNAct(int(in_channels), base, kernel_size=3, stride=2, act="relu"),  # /2
            ConvBNAct(base, base, kernel_size=3, stride=2, act="relu"),  # /4
        )

        enc: list[nn.Module] = []
        ch = base
        for _ in range(lv - 1):
            enc.append(_EncStage(ch, ch * 2, depth=d))
            ch *= 2
        self.enc = nn.ModuleList(enc)

        dec: list[nn.Module] = []
        for _ in range(lv - 1):
            dec.append(_DecStage(ch, ch // 2, depth=d))
            ch //= 2
        self.dec = nn.ModuleList(dec)

        self.drop = nn.Dropout2d(p=float(dropout))
        self.out = nn.Conv2d(base, nc, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = check_nchw(x)
        inp_hw = x.shape[-2:]

        skips: list[torch.Tensor] = []
        x = self.stem(x)
        skips.append(x)
        for stage in self.enc:
            x = stage(x)
            skips.append(x)

        for stage, skip in zip(self.dec, reversed(skips[:-1]), strict=True):
            x = stage(x, skip)

        x = self.drop(x)
        logits = self.out(x)
        return F.interpolate(logits, size=inp_hw, mode="nearest")


_VARIANTS: dict[str, dict] = {
    "linknet_tiny": {"base_channels": 16, "levels": 3, "depth": 1, "dropout": 0.0},
    "linknet_small": {"base_channels": 24, "levels": 4, "depth": 2, "dropout": 0.0},
    "linknet_base": {"base_channels": 32, "levels": 4, "depth": 2, "dropout": 0.1},
}


def build_linknet_segmenter(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "linknet_small",
    dropout: float | None = None,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown LinkNet variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    p = float(spec["dropout"]) if dropout is None else float(dropout)
    return LinkNet(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        base_channels=int(spec["base_channels"]),
        levels=int(spec["levels"]),
        depth=int(spec["depth"]),
        dropout=float(p),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_linknet_segmenter(in_channels=3, num_classes=4, variant="linknet_tiny")
    y = m(x)
    print("linknet_tiny", tuple(y.shape))
    loss = y.mean()
    loss.backward()
    print("ok")
