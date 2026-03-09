
import torch
from torch import nn

from dlhub.vision.backbones._blocks import ConvBNAct, scale_channels


class Fire(nn.Module):
    """SqueezeNet fire module (toy)."""

    def __init__(self, in_ch: int, squeeze_ch: int, expand_ch: int) -> None:
        super().__init__()
        self.squeeze = ConvBNAct(int(in_ch), int(squeeze_ch), kernel_size=1, stride=1, act="relu")
        self.expand1 = ConvBNAct(int(squeeze_ch), int(expand_ch), kernel_size=1, stride=1, act="relu")
        self.expand3 = ConvBNAct(int(squeeze_ch), int(expand_ch), kernel_size=3, stride=1, padding=1, act="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.squeeze(x)
        return torch.cat([self.expand1(x), self.expand3(x)], dim=1)


class SqueezeDetDetector(nn.Module):
    """SqueezeDet-style single-shot detector (toy-first).

    Forward returns:
    - cls_logits: (B, A*C, H/16, W/16)
    - bbox_deltas: (B, A*4, H/16, W/16)
    """

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        stem_channels: int = 32,
        fire_channels: int = 64,
        depth: int = 3,
        num_anchors: int = 9,
    ) -> None:
        super().__init__()
        c_in = int(in_channels)
        nc = int(num_classes)
        na = int(num_anchors)
        d = int(depth)
        if nc <= 0:
            raise ValueError("num_classes must be > 0")
        if na <= 0:
            raise ValueError("num_anchors must be > 0")
        if d <= 0:
            raise ValueError("depth must be > 0")

        stem = int(stem_channels)
        f = int(fire_channels)
        self.net = nn.Sequential(
            ConvBNAct(c_in, stem, kernel_size=3, stride=2, act="relu"),  # /2
            ConvBNAct(stem, stem, kernel_size=3, stride=2, act="relu"),  # /4
            ConvBNAct(stem, f, kernel_size=3, stride=2, act="relu"),  # /8
            ConvBNAct(f, f, kernel_size=3, stride=2, act="relu"),  # /16
        )
        fire_in = f
        fires: list[nn.Module] = []
        for _ in range(d):
            fires.append(Fire(fire_in, squeeze_ch=max(8, f // 4), expand_ch=max(8, f // 4)))
            fire_in = max(8, f // 4) * 2
        self.fires = nn.Sequential(*fires)

        out_ch = fire_in
        self.cls = nn.Conv2d(out_ch, na * nc, kernel_size=3, padding=1)
        self.box = nn.Conv2d(out_ch, na * 4, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = x.to(torch.float32)
        if x.ndim != 4:
            raise ValueError(f"Expected input shape (B, C, H, W), got {tuple(x.shape)}")
        f = self.fires(self.net(x))
        return {"cls_logits": self.cls(f), "bbox_deltas": self.box(f)}


_VARIANTS: dict[str, dict] = {
    "squeezedet_tiny": {"stem": 24, "fire": 48, "depth": 2, "anchors": 6},
    "squeezedet_small": {"stem": 32, "fire": 64, "depth": 3, "anchors": 9},
    "squeezedet_base": {"stem": 48, "fire": 96, "depth": 4, "anchors": 9},
}


def build_squeezedet_detector(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "squeezedet_tiny",
    width_mult: float = 1.0,
    num_anchors: int | None = None,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown SqueezeDet variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    stem = scale_channels(int(spec["stem"]), float(width_mult), min_ch=16, divisor=8)
    fire = scale_channels(int(spec["fire"]), float(width_mult), min_ch=16, divisor=8)
    na = int(spec["anchors"]) if num_anchors is None else int(num_anchors)
    return SqueezeDetDetector(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        stem_channels=int(stem),
        fire_channels=int(fire),
        depth=int(spec["depth"]),
        num_anchors=int(na),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 128, 128)
    m = build_squeezedet_detector(in_channels=3, num_classes=3, variant="squeezedet_tiny", width_mult=1.0)
    out = m(x)
    print("squeezedet_tiny", {k: tuple(v.shape) for k, v in out.items()})
    loss = sum(v.mean() for v in out.values())
    loss.backward()
    print("ok")

