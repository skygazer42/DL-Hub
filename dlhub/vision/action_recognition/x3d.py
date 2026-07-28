"""X3D - compact-first video action classifier (efficient 3D conv).

Reference:
- "X3D: Expanding Architectures for Efficient Video Recognition" (CVPR 2020)

Compact interpretation:
- A small stack of inverted-residual style 3D blocks with depthwise 3D conv.
- Optional 3D squeeze-excite for channel gating.
"""

import torch
import torch.nn.functional as F
from torch import nn

from dlhub.vision.backbones._blocks import scale_channels

from ._common import check_video_input


class Conv3dBNAct(nn.Sequential):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        *,
        kernel_size: int | tuple[int, int, int],
        stride: int | tuple[int, int, int] = 1,
        padding: int | tuple[int, int, int] | None = None,
        groups: int = 1,
        act: str = "silu",
    ) -> None:
        if padding is None:
            if isinstance(kernel_size, tuple):
                padding = tuple(int(k) // 2 for k in kernel_size)
            else:
                padding = int(kernel_size) // 2

        act_name = str(act).lower().strip()
        if act_name in {"silu", "swish"}:
            act_layer: nn.Module = nn.SiLU(inplace=True)
        elif act_name == "relu":
            act_layer = nn.ReLU(inplace=True)
        else:
            raise ValueError(f"Unknown activation: {act!r}")

        super().__init__(
            nn.Conv3d(
                int(in_ch),
                int(out_ch),
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=int(groups),
                bias=False,
            ),
            nn.BatchNorm3d(int(out_ch)),
            act_layer,
        )


class SqueezeExcite3D(nn.Module):
    def __init__(self, channels: int, *, se_ratio: float = 0.25) -> None:
        super().__init__()
        c = int(channels)
        hidden = max(8, int(round(c * float(se_ratio))))
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc1 = nn.Conv3d(c, hidden, kernel_size=1, bias=True)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv3d(hidden, c, kernel_size=1, bias=True)
        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.pool(x)
        s = self.act(self.fc1(s))
        s = self.gate(self.fc2(s))
        return x * s


class X3DBlock(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        stride: tuple[int, int, int],
        expand_ratio: float,
        se_ratio: float,
    ) -> None:
        super().__init__()
        c_in = int(in_channels)
        c_out = int(out_channels)
        st, sh, sw = map(int, stride)
        if st <= 0 or sh <= 0 or sw <= 0:
            raise ValueError("stride must be positive")

        hidden = max(8, int(round(c_in * float(expand_ratio))))
        self.use_res = (st, sh, sw) == (1, 1, 1) and c_in == c_out

        layers: list[nn.Module] = []
        if hidden != c_in:
            layers.append(Conv3dBNAct(c_in, hidden, kernel_size=1, stride=1, padding=0, act="silu"))
        # depthwise 3D conv
        layers.append(
            Conv3dBNAct(
                hidden,
                hidden,
                kernel_size=3,
                stride=(st, sh, sw),
                padding=1,
                groups=hidden,
                act="silu",
            )
        )
        if float(se_ratio) > 0:
            layers.append(SqueezeExcite3D(hidden, se_ratio=float(se_ratio)))
        # projection (linear)
        layers.append(nn.Conv3d(hidden, c_out, kernel_size=1, bias=False))
        layers.append(nn.BatchNorm3d(c_out))

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.block(x)
        if self.use_res:
            return x + y
        return y


class X3DVideoClassifier(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        width: int,
        depth: int,
        expand_ratio: float,
        se_ratio: float,
        dropout: float,
    ) -> None:
        super().__init__()
        c_in = int(in_channels)
        w = int(width)
        d = int(depth)
        if c_in <= 0:
            raise ValueError("in_channels must be > 0")
        if w < 8:
            raise ValueError("width must be >= 8")
        if d <= 0:
            raise ValueError("depth must be > 0")

        self.stem = nn.Sequential(
            Conv3dBNAct(c_in, w, kernel_size=3, stride=(1, 2, 2), padding=1, act="silu"),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
        )

        blocks: list[nn.Module] = []
        ch = w
        for i in range(d):
            out = ch if i == 0 else ch * 2
            stride = (
                (1, 1, 1)
                if i == 0
                else (2 if i == 1 else 1, 2 if i == 1 else 1, 2 if i == 1 else 1)
            )
            # Only downsample on i==1 to keep the compact fast.
            if i != 1:
                stride = (1, 1, 1)
            blocks.append(
                X3DBlock(
                    in_channels=ch,
                    out_channels=out,
                    stride=tuple(int(s) for s in stride),
                    expand_ratio=float(expand_ratio),
                    se_ratio=float(se_ratio),
                )
            )
            ch = out

        self.blocks = nn.Sequential(*blocks)
        self.dropout = nn.Dropout(float(dropout))
        self.classifier = nn.Linear(int(ch), int(num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = check_video_input(x)
        x = self.stem(x)
        x = self.blocks(x)
        x = F.adaptive_avg_pool3d(x, (1, 1, 1)).flatten(1)
        x = self.dropout(x)
        return self.classifier(x)


_VARIANTS: dict[str, dict] = {
    "x3d_tiny": {"width": 24, "depth": 2, "expand": 2.0, "se": 0.25},
    "x3d_small": {"width": 32, "depth": 3, "expand": 2.5, "se": 0.25},
    "x3d_base": {"width": 48, "depth": 4, "expand": 3.0, "se": 0.25},
}


def build_x3d_video_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "x3d_small",
    width_mult: float = 1.0,
    dropout: float = 0.1,
    # Unused but kept for zoo/script signature consistency.
    frames: int = 8,
    image_size: int = 64,
) -> nn.Module:
    del frames, image_size
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown X3D variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    width = scale_channels(int(spec["width"]), float(width_mult), min_ch=8, divisor=8)
    return X3DVideoClassifier(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        width=int(width),
        depth=int(spec["depth"]),
        expand_ratio=float(spec["expand"]),
        se_ratio=float(spec["se"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 8, 64, 64)
    m = build_x3d_video_classifier(
        in_channels=3, num_classes=6, variant="x3d_tiny", width_mult=0.5, dropout=0.0
    )
    y = m(x)
    print("x3d_tiny", tuple(y.shape))
    loss = y.mean()
    loss.backward()
    print("ok")
