import torch
import torch.nn.functional as F
from torch import nn

from dlhub.vision.backbones._blocks import ConvBNAct, scale_channels


class _SimpleBackboneLowHigh(nn.Module):
    """Backbone that returns (low_level, high_level) features for DeepLabV3+."""

    def __init__(
        self,
        *,
        in_channels: int,
        stem_channels: int,
        low_channels: int,
        high_channels: int,
        depth: int,
    ) -> None:
        super().__init__()
        c_in = int(in_channels)
        stem = int(stem_channels)
        low = int(low_channels)
        high = int(high_channels)
        d = int(depth)
        if d <= 0:
            raise ValueError("depth must be > 0")

        self.stem = nn.Sequential(
            ConvBNAct(c_in, stem, kernel_size=3, stride=2, act="relu"),  # /2
            ConvBNAct(stem, low, kernel_size=3, stride=2, act="relu"),  # /4
        )

        def make_stage(in_ch: int, out_ch: int) -> nn.Sequential:
            layers: list[nn.Module] = [
                ConvBNAct(in_ch, out_ch, kernel_size=3, stride=2, act="relu")
            ]
            for _ in range(d - 1):
                layers.append(ConvBNAct(out_ch, out_ch, kernel_size=3, stride=1, act="relu"))
            return nn.Sequential(*layers)

        self.stage8 = make_stage(low, max(low, low * 2))  # /8
        self.stage16 = make_stage(max(low, low * 2), high)  # /16

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        low = self.stem(x)  # /4
        x = self.stage8(low)  # /8
        high = self.stage16(x)  # /16
        return low, high


class ASPP(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, *, rates: tuple[int, int, int]) -> None:
        super().__init__()
        c_in = int(in_ch)
        c_out = int(out_ch)
        r1, r2, r3 = (int(r) for r in rates)

        self.b0 = ConvBNAct(c_in, c_out, kernel_size=1, stride=1, padding=0, act="relu")
        # Use explicit dilated conv branches.
        self.d1 = nn.Conv2d(c_in, c_out, kernel_size=3, padding=r1, dilation=r1, bias=False)
        self.d2 = nn.Conv2d(c_in, c_out, kernel_size=3, padding=r2, dilation=r2, bias=False)
        self.d3 = nn.Conv2d(c_in, c_out, kernel_size=3, padding=r3, dilation=r3, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.bn2 = nn.BatchNorm2d(c_out)
        self.bn3 = nn.BatchNorm2d(c_out)
        self.act = nn.ReLU(inplace=True)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_proj = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )

        self.project = nn.Sequential(
            nn.Conv2d(c_out * 5, c_out, kernel_size=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        b0 = self.b0(x)
        # dilated conv branches
        b1 = self.act(self.bn1(self.d1(x)))
        b2 = self.act(self.bn2(self.d2(x)))
        b3 = self.act(self.bn3(self.d3(x)))

        gp = self.pool(x)
        gp = self.pool_proj(gp)
        gp = F.interpolate(gp, size=(h, w), mode="nearest")

        y = torch.cat([b0, b1, b2, b3, gp], dim=1)
        return self.project(y)


class DeepLabV3Plus(nn.Module):
    """DeepLabV3+ (compact-first, pure torch).

    Forward: (B, C, H, W) -> logits (B, num_classes, H, W)
    """

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        stem_channels: int = 32,
        low_channels: int = 48,
        high_channels: int = 128,
        backbone_depth: int = 2,
        aspp_channels: int = 128,
        aspp_rates: tuple[int, int, int] = (2, 4, 6),
        decoder_channels: int = 96,
        low_proj_channels: int = 32,
    ) -> None:
        super().__init__()
        c_in = int(in_channels)
        nc = int(num_classes)
        if nc <= 0:
            raise ValueError("num_classes must be > 0")

        self.backbone = _SimpleBackboneLowHigh(
            in_channels=c_in,
            stem_channels=int(stem_channels),
            low_channels=int(low_channels),
            high_channels=int(high_channels),
            depth=int(backbone_depth),
        )
        self.aspp = ASPP(int(high_channels), int(aspp_channels), rates=tuple(aspp_rates))

        self.low_proj = nn.Sequential(
            nn.Conv2d(int(low_channels), int(low_proj_channels), kernel_size=1, bias=False),
            nn.BatchNorm2d(int(low_proj_channels)),
            nn.ReLU(inplace=True),
        )

        dec_in = int(aspp_channels) + int(low_proj_channels)
        self.decoder = nn.Sequential(
            ConvBNAct(dec_in, int(decoder_channels), kernel_size=3, stride=1, act="relu"),
            ConvBNAct(
                int(decoder_channels), int(decoder_channels), kernel_size=3, stride=1, act="relu"
            ),
        )
        self.classifier = nn.Conv2d(int(decoder_channels), nc, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        if x.ndim != 4:
            raise ValueError(f"Expected input shape (B, C, H, W), got {tuple(x.shape)}")

        inp_h, inp_w = x.shape[-2:]
        low, high = self.backbone(x)
        y = self.aspp(high)
        y = F.interpolate(y, size=low.shape[-2:], mode="bilinear", align_corners=False)
        low_p = self.low_proj(low)
        y = self.decoder(torch.cat([y, low_p], dim=1))
        logits = self.classifier(y)
        return F.interpolate(logits, size=(inp_h, inp_w), mode="bilinear", align_corners=False)


_VARIANTS: dict[str, dict] = {
    "deeplabv3p_tiny": {
        "stem": 24,
        "low": 32,
        "high": 96,
        "depth": 1,
        "aspp": 96,
        "dec": 64,
        "lowproj": 24,
    },
    "deeplabv3p_small": {
        "stem": 32,
        "low": 48,
        "high": 128,
        "depth": 2,
        "aspp": 128,
        "dec": 96,
        "lowproj": 32,
    },
    "deeplabv3p_base": {
        "stem": 48,
        "low": 64,
        "high": 192,
        "depth": 3,
        "aspp": 192,
        "dec": 128,
        "lowproj": 48,
    },
}


def build_deeplabv3plus_segmenter(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "deeplabv3p_small",
    width_mult: float = 1.0,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown DeepLabV3+ variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]

    stem = scale_channels(int(spec["stem"]), float(width_mult), min_ch=16, divisor=8)
    low = scale_channels(int(spec["low"]), float(width_mult), min_ch=16, divisor=8)
    high = scale_channels(int(spec["high"]), float(width_mult), min_ch=16, divisor=8)
    aspp = scale_channels(int(spec["aspp"]), float(width_mult), min_ch=16, divisor=8)
    dec = scale_channels(int(spec["dec"]), float(width_mult), min_ch=16, divisor=8)
    lowproj = scale_channels(int(spec["lowproj"]), float(width_mult), min_ch=16, divisor=8)

    return DeepLabV3Plus(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        stem_channels=int(stem),
        low_channels=int(low),
        high_channels=int(high),
        backbone_depth=int(spec["depth"]),
        aspp_channels=int(aspp),
        aspp_rates=(2, 4, 6),
        decoder_channels=int(dec),
        low_proj_channels=int(lowproj),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_deeplabv3plus_segmenter(in_channels=3, num_classes=2, variant="deeplabv3p_tiny")
    y = m(x)
    print("deeplabv3p_tiny", tuple(y.shape))
    loss = y.mean()
    loss.backward()
    print("ok")
