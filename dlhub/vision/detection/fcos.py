from __future__ import annotations

import torch
from torch import nn

from dlhub.vision.backbones._blocks import ConvBNAct, scale_channels


class _ConvTower(nn.Module):
    def __init__(self, channels: int, *, num_convs: int = 4) -> None:
        super().__init__()
        c = int(channels)
        n = int(num_convs)
        if n <= 0:
            raise ValueError("num_convs must be > 0")
        self.net = nn.Sequential(*[ConvBNAct(c, c, kernel_size=3, stride=1, act="relu") for _ in range(n)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _TinyBackbone(nn.Module):
    """A tiny conv backbone that outputs a stride-4 feature map."""

    def __init__(
        self,
        *,
        in_channels: int,
        stem_channels: int,
        out_channels: int,
        depth: int,
    ) -> None:
        super().__init__()
        c_in = int(in_channels)
        stem = int(stem_channels)
        out = int(out_channels)
        d = int(depth)
        if d <= 0:
            raise ValueError("depth must be > 0")

        layers: list[nn.Module] = [
            ConvBNAct(c_in, stem, kernel_size=3, stride=2, act="relu"),
            ConvBNAct(stem, out, kernel_size=3, stride=2, act="relu"),
        ]
        for _ in range(d):
            layers.append(ConvBNAct(out, out, kernel_size=3, stride=1, act="relu"))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FCOSDetector(nn.Module):
    """FCOS-style anchor-free detector (toy-first, single-level).

    Output (raw):
    - cls_logits: (B, num_classes, H/4, W/4)
    - reg: (B, 4, H/4, W/4)  # l,t,r,b distances (non-negative after ReLU)
    - centerness: optional (B, 1, H/4, W/4)
    """

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        stem_channels: int = 32,
        hidden_channels: int = 64,
        backbone_depth: int = 2,
        head_convs: int = 4,
        with_centerness: bool = True,
    ) -> None:
        super().__init__()
        c_in = int(in_channels)
        nc = int(num_classes)
        if nc <= 0:
            raise ValueError("num_classes must be > 0")

        self.stride = 4
        self.backbone = _TinyBackbone(
            in_channels=c_in,
            stem_channels=int(stem_channels),
            out_channels=int(hidden_channels),
            depth=int(backbone_depth),
        )

        self.cls_tower = _ConvTower(int(hidden_channels), num_convs=int(head_convs))
        self.reg_tower = _ConvTower(int(hidden_channels), num_convs=int(head_convs))

        self.cls_logits = nn.Conv2d(int(hidden_channels), nc, kernel_size=3, padding=1)
        self.reg_pred = nn.Conv2d(int(hidden_channels), 4, kernel_size=3, padding=1)
        self.with_centerness = bool(with_centerness)
        self.centerness = nn.Conv2d(int(hidden_channels), 1, kernel_size=3, padding=1) if self.with_centerness else None

        # A tiny init that helps stability (optional).
        nn.init.normal_(self.cls_logits.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.cls_logits.bias, 0.0)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = x.to(torch.float32)
        if x.ndim != 4:
            raise ValueError(f"Expected input shape (B, C, H, W), got {tuple(x.shape)}")

        feats = self.backbone(x)
        cls_feats = self.cls_tower(feats)
        reg_feats = self.reg_tower(feats)

        cls_logits = self.cls_logits(cls_feats)
        reg = torch.relu(self.reg_pred(reg_feats))
        out = {"cls_logits": cls_logits, "reg": reg}
        if self.centerness is not None:
            out["centerness"] = self.centerness(reg_feats)
        return out


_VARIANTS: dict[str, dict] = {
    "fcos_tiny": {"stem": 24, "hidden": 32, "backbone_depth": 1, "head_convs": 2, "centerness": True},
    "fcos_small": {"stem": 32, "hidden": 64, "backbone_depth": 2, "head_convs": 3, "centerness": True},
    "fcos_base": {"stem": 48, "hidden": 96, "backbone_depth": 3, "head_convs": 4, "centerness": True},
    "fcos_nocenter": {"stem": 32, "hidden": 64, "backbone_depth": 2, "head_convs": 3, "centerness": False},
}


def build_fcos_detector(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "fcos_tiny",
    width_mult: float = 1.0,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown FCOS variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    stem = scale_channels(int(spec["stem"]), float(width_mult), min_ch=16, divisor=8)
    hidden = scale_channels(int(spec["hidden"]), float(width_mult), min_ch=16, divisor=8)
    return FCOSDetector(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        stem_channels=int(stem),
        hidden_channels=int(hidden),
        backbone_depth=int(spec["backbone_depth"]),
        head_convs=int(spec["head_convs"]),
        with_centerness=bool(spec["centerness"]),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_fcos_detector(in_channels=3, num_classes=3, variant="fcos_tiny", width_mult=1.0)
    out = m(x)
    print("fcos_tiny", {k: tuple(v.shape) for k, v in out.items()})
    loss = sum(v.mean() for v in out.values())
    loss.backward()
    print("ok")

