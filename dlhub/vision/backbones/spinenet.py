from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from dlhub.vision.backbones._blocks import ConvBNAct, DropPath, GlobalAvgPoolHead, scale_channels


@dataclass(frozen=True)
class SpineNodeSpec:
    a: int
    b: int
    out_level: int  # p2..p7 style level where stride=2**level
    out_ch: int


class ResampleToLevel(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, *, in_level: int, out_level: int) -> None:
        super().__init__()
        self.in_level = int(in_level)
        self.out_level = int(out_level)
        self.proj = nn.Sequential(
            nn.Conv2d(int(in_ch), int(out_ch), kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(int(out_ch)),
        )

    def forward(self, x: torch.Tensor, *, out_hw: tuple[int, int]) -> torch.Tensor:
        h, w = int(out_hw[0]), int(out_hw[1])
        if x.shape[-2:] != (h, w):
            x = F.interpolate(x, size=(h, w), mode="nearest")
        return self.proj(x)


class SpineBlock(nn.Module):
    def __init__(self, channels: int, *, drop_path: float = 0.0) -> None:
        super().__init__()
        c = int(channels)
        self.conv1 = ConvBNAct(c, c, kernel_size=3, stride=1, act="silu")
        self.conv2 = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c),
        )
        self.dp = DropPath(float(drop_path))
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.dp(x)
        return self.act(x + identity)


class SpineNetClassifier(nn.Module):
    """SpineNet-ish searched feature pyramid (simplified for classification).

    This is not a detection FPN; it builds a small searched-like computation graph
    across multiple feature levels and finishes with a classifier head.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        dims: tuple[int, int, int, int] = (64, 128, 256, 512),
        nodes: tuple[SpineNodeSpec, ...],
        width_mult: float = 1.0,
        drop_path: float = 0.1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        dims = tuple(scale_channels(int(d), float(width_mult), min_ch=16, divisor=8) for d in dims)

        self.stem = nn.Sequential(
            ConvBNAct(int(in_channels), dims[0], kernel_size=3, stride=2, act="silu"),
            ConvBNAct(dims[0], dims[0], kernel_size=3, stride=2, act="silu"),  # stride 4 (p2)
        )
        self.p3 = ConvBNAct(dims[0], dims[1], kernel_size=3, stride=2, act="silu")  # stride 8 (p3)
        self.p4 = ConvBNAct(dims[1], dims[2], kernel_size=3, stride=2, act="silu")  # stride 16 (p4)
        self.p5 = ConvBNAct(dims[2], dims[3], kernel_size=3, stride=2, act="silu")  # stride 32 (p5)

        specs = tuple(nodes)
        self.node_specs = specs

        # Build resamplers/blocks with deterministic channel bookkeeping.
        feat_levels: list[int] = [2, 3, 4, 5]
        feat_channels: list[int] = [int(dims[0]), int(dims[1]), int(dims[2]), int(dims[3])]

        total_nodes = len(specs)
        dp_rates = torch.linspace(0.0, float(drop_path), steps=max(1, total_nodes)).tolist()
        dp_iter = iter(dp_rates)

        resamplers: list[nn.Module] = []
        blocks: list[nn.Module] = []
        for spec in specs:
            a_idx, b_idx = int(spec.a), int(spec.b)
            if a_idx < 0 or a_idx >= len(feat_channels) or b_idx < 0 or b_idx >= len(feat_channels):
                raise ValueError(
                    f"Invalid SpineNet node inputs: a={a_idx}, b={b_idx}, available={len(feat_channels)}"
                )

            out_level = int(spec.out_level)
            out_ch = int(spec.out_ch)

            resamplers.append(
                ResampleToLevel(
                    feat_channels[a_idx],
                    out_ch,
                    in_level=int(feat_levels[a_idx]),
                    out_level=int(out_level),
                )
            )
            resamplers.append(
                ResampleToLevel(
                    feat_channels[b_idx],
                    out_ch,
                    in_level=int(feat_levels[b_idx]),
                    out_level=int(out_level),
                )
            )
            blocks.append(SpineBlock(out_ch, drop_path=float(next(dp_iter, 0.0))))

            feat_levels.append(int(out_level))
            feat_channels.append(int(out_ch))

        self._resamplers = nn.ModuleList(resamplers)
        self._blocks = nn.ModuleList(blocks)

        # Final head takes the highest-level feature that exists (largest level).
        best_level = max(int(level) for level in feat_levels)
        best_idx = max(i for i, level in enumerate(feat_levels) if int(level) == best_level)
        head_ch = int(feat_channels[int(best_idx)])
        self.head = GlobalAvgPoolHead(head_ch, int(num_classes), dropout=float(dropout))

    def _level_hw(self, x: torch.Tensor, level: int) -> tuple[int, int]:
        # level=2 -> stride=4, level=3 -> stride=8, ...
        h, w = int(x.shape[-2]), int(x.shape[-1])
        stride = 2 ** int(level)
        return max(1, h // stride), max(1, w // stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)

        p2 = self.stem(x)
        p3 = self.p3(p2)
        p4 = self.p4(p3)
        p5 = self.p5(p4)

        feats: list[torch.Tensor] = [p2, p3, p4, p5]
        levels: list[int] = [2, 3, 4, 5]

        for i, spec in enumerate(self.node_specs):
            a_idx, b_idx = int(spec.a), int(spec.b)
            out_level = int(spec.out_level)

            out_hw = self._level_hw(x, out_level)
            ra = self._resamplers[2 * i](feats[a_idx], out_hw=out_hw)
            rb = self._resamplers[2 * i + 1](feats[b_idx], out_hw=out_hw)
            y = ra + rb
            y = self._blocks[i](y)
            feats.append(y)
            levels.append(out_level)

        # pick the highest-level feature (largest level number)
        best_level = max(int(level) for level in levels)
        best_idx = max(i for i, level in enumerate(levels) if int(level) == best_level)
        return self.head(feats[int(best_idx)])


_VARIANTS: dict[str, dict] = {
    "spinenet_tiny": {
        "dims": (48, 96, 192, 384),
        "nodes": (
            SpineNodeSpec(1, 2, out_level=4, out_ch=192),
            SpineNodeSpec(0, 4, out_level=3, out_ch=128),
            SpineNodeSpec(2, 3, out_level=5, out_ch=256),
            SpineNodeSpec(5, 6, out_level=5, out_ch=256),
        ),
    },
    "spinenet_small": {
        "dims": (64, 128, 256, 512),
        "nodes": (
            SpineNodeSpec(1, 2, out_level=4, out_ch=256),
            SpineNodeSpec(0, 4, out_level=3, out_ch=160),
            SpineNodeSpec(2, 3, out_level=5, out_ch=384),
            SpineNodeSpec(5, 6, out_level=5, out_ch=384),
            SpineNodeSpec(7, 4, out_level=4, out_ch=256),
            SpineNodeSpec(8, 3, out_level=5, out_ch=384),
        ),
    },
    "spinenet_base": {
        "dims": (80, 160, 320, 640),
        "nodes": (
            SpineNodeSpec(1, 2, out_level=4, out_ch=320),
            SpineNodeSpec(0, 4, out_level=3, out_ch=192),
            SpineNodeSpec(2, 3, out_level=5, out_ch=512),
            SpineNodeSpec(5, 6, out_level=5, out_ch=512),
            SpineNodeSpec(7, 4, out_level=4, out_ch=320),
            SpineNodeSpec(8, 3, out_level=5, out_ch=512),
            SpineNodeSpec(9, 8, out_level=4, out_ch=320),
            SpineNodeSpec(10, 6, out_level=5, out_ch=512),
        ),
    },
}


def build_spinenet_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "spinenet_tiny",
    width_mult: float = 1.0,
    drop_path: float = 0.1,
    dropout: float = 0.1,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown SpineNet variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    w = float(width_mult)
    scaled_nodes = tuple(
        SpineNodeSpec(
            int(n.a),
            int(n.b),
            out_level=int(n.out_level),
            out_ch=scale_channels(int(n.out_ch), w, min_ch=16, divisor=8),
        )
        for n in spec["nodes"]
    )
    return SpineNetClassifier(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        dims=tuple(map(int, spec["dims"])),
        nodes=scaled_nodes,
        width_mult=float(width_mult),
        drop_path=float(drop_path),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_spinenet_classifier(
        in_channels=3, num_classes=10, variant="spinenet_tiny", width_mult=0.5
    )
    y = m(x)
    print("spinenet_tiny", tuple(y.shape))
