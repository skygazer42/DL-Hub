from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from dlhub.vision.backbones._blocks import ConvBNAct, GlobalAvgPoolHead, scale_channels


class DeformableConv2d(nn.Module):
    """A minimal Deformable Conv2d (v1) implementation using grid_sample.

    This implementation is designed for correctness and simplicity, not speed.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        *,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int | None = None,
    ) -> None:
        super().__init__()
        k = int(kernel_size)
        if padding is None:
            padding = k // 2
        self.k = k
        self.stride = int(stride)
        self.padding = int(padding)

        self.offset = nn.Conv2d(int(in_ch), 2 * k * k, kernel_size=3, stride=self.stride, padding=1, bias=True)
        self.weight = nn.Parameter(torch.randn(int(out_ch), int(in_ch), k, k) * 0.02)
        self.bias = nn.Parameter(torch.zeros(int(out_ch)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        b, c, h, w = x.shape
        k = self.k
        s = self.stride
        p = self.padding

        # Offsets are predicted on the unpadded input; output spatial size follows conv formula.
        offsets = self.offset(x)  # (B, 2*k*k, Hout, Wout)
        _, _, h_out, w_out = offsets.shape

        x_pad = F.pad(x, (p, p, p, p))
        h_pad, w_pad = x_pad.shape[-2:]

        # Base coords for the top-left corner of each receptive field in padded coords.
        base_y = torch.arange(h_out, device=x.device, dtype=x.dtype) * s
        base_x = torch.arange(w_out, device=x.device, dtype=x.dtype) * s
        yy, xx = torch.meshgrid(base_y, base_x, indexing="ij")  # (Hout, Wout)
        yy = yy[None, :, :].expand(b, -1, -1)  # (B, Hout, Wout)
        xx = xx[None, :, :].expand(b, -1, -1)

        out = torch.zeros(b, self.weight.shape[0], h_out, w_out, device=x.device, dtype=x.dtype)
        for idx in range(k * k):
            ky = idx // k
            kx = idx % k
            dy = offsets[:, 2 * idx + 0]
            dx = offsets[:, 2 * idx + 1]
            sample_y = yy + float(ky) + dy
            sample_x = xx + float(kx) + dx

            # Normalize to [-1,1]
            gy = (sample_y / max(1.0, float(h_pad - 1))) * 2.0 - 1.0
            gx = (sample_x / max(1.0, float(w_pad - 1))) * 2.0 - 1.0
            grid = torch.stack([gx, gy], dim=-1)  # (B, Hout, Wout, 2)
            sampled = F.grid_sample(x_pad, grid, mode="bilinear", padding_mode="zeros", align_corners=False)  # (B,C,Hout,Wout)

            w_ij = self.weight[:, :, ky, kx].unsqueeze(-1).unsqueeze(-1)  # (O, C, 1, 1)
            out = out + F.conv2d(sampled, w_ij, bias=None, stride=1, padding=0)

        out = out + self.bias.view(1, -1, 1, 1)
        return out


class DCNBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, *, stride: int = 1) -> None:
        super().__init__()
        self.dcn = DeformableConv2d(int(in_ch), int(out_ch), kernel_size=3, stride=int(stride), padding=1)
        self.bn = nn.BatchNorm2d(int(out_ch))
        self.act = nn.ReLU(inplace=True)
        self.pw = ConvBNAct(int(out_ch), int(out_ch), kernel_size=1, stride=1, padding=0, act="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dcn(x)
        x = self.act(self.bn(x))
        return self.pw(x)


class DCNClassifier(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        channels: tuple[int, int, int, int] = (64, 128, 256, 512),
        widths: tuple[int, int, int, int] | None = None,
        width_mult: float = 1.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        chs = tuple(scale_channels(int(c), float(width_mult)) for c in channels)
        self.stem = nn.Sequential(
            ConvBNAct(int(in_channels), chs[0], kernel_size=3, stride=2, act="relu"),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.stage1 = DCNBlock(chs[0], chs[0], stride=1)
        self.stage2 = DCNBlock(chs[0], chs[1], stride=2)
        self.stage3 = DCNBlock(chs[1], chs[2], stride=2)
        self.stage4 = DCNBlock(chs[2], chs[3], stride=2)
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
    "dcn_tiny": {"channels": (48, 96, 192, 384)},
    "dcn_base": {"channels": (64, 128, 256, 512)},
}


def build_dcn_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "dcn_base",
    width_mult: float = 1.0,
    dropout: float = 0.1,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown DCN variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    return DCNClassifier(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        channels=tuple(map(int, spec["channels"])),
        width_mult=float(width_mult),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(1, 3, 64, 64)
    m = build_dcn_classifier(in_channels=3, num_classes=10, variant="dcn_tiny", width_mult=0.5)
    y = m(x)
    print("dcn_tiny", tuple(y.shape))

