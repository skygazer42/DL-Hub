import torch
import torch.nn.functional as F
from torch import nn


class MaskedConv2d(nn.Module):
    """PixelCNN-style masked convolution (mask type 'A' or 'B').

    Mask type:
    - 'A': excludes the center pixel (strict blind-spot at first layer)
    - 'B': includes the center pixel (safe after an 'A' layer)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 3,
        mask_type: str = "B",
    ) -> None:
        super().__init__()
        k = int(kernel_size)
        if k < 3 or k % 2 == 0:
            raise ValueError("kernel_size must be odd and >= 3")

        mt = str(mask_type).upper().strip()
        if mt not in {"A", "B"}:
            raise ValueError(f"mask_type must be 'A' or 'B', got: {mask_type!r}")

        p = k // 2
        self.conv = nn.Conv2d(
            int(in_channels), int(out_channels), kernel_size=k, padding=p, bias=True
        )

        mask = torch.ones_like(self.conv.weight)
        cy = p
        cx = p

        mask[:, :, cy + 1 :, :] = 0  # rows below center
        if mt == "A":
            mask[:, :, cy, cx:] = 0  # exclude center
        else:
            mask[:, :, cy, cx + 1 :] = 0  # include center

        self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.conv.weight * self.mask
        return F.conv2d(x, w, bias=self.conv.bias, stride=1, padding=self.conv.padding)


class GatedMaskedResBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        c = int(channels)
        self.conv_f = MaskedConv2d(c, c, kernel_size=3, mask_type="B")
        self.conv_g = MaskedConv2d(c, c, kernel_size=3, mask_type="B")
        self.out = nn.Conv2d(c, c, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.conv_f(x)
        g = self.conv_g(x)
        y = torch.tanh(f) * torch.sigmoid(g)
        y = self.out(y)
        return F.relu(x + y, inplace=True)


class GatedPixelCNNBlindSpotBackbone(nn.Module):
    def __init__(self, *, in_channels: int, width: int = 24, depth: int = 6) -> None:
        super().__init__()
        c_in = int(in_channels)
        w0 = int(width)
        d = int(depth)
        if c_in <= 0:
            raise ValueError("in_channels must be > 0")
        if w0 < 8:
            raise ValueError("width must be >= 8")
        if d <= 0:
            raise ValueError("depth must be > 0")

        self.in_conv = MaskedConv2d(c_in, w0, kernel_size=5, mask_type="A")
        self.blocks = nn.Sequential(*[GatedMaskedResBlock(w0) for _ in range(d)])
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.in_conv(x))
        return self.blocks(x)


class GatedPixelCNNBSN(nn.Module):
    """Gated PixelCNN masked-conv + rotation fusion blind-spot denoiser (compact-first).

    Same overall idea as `PixelCNNBSN`, but uses gated masked residual blocks.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        width: int = 24,
        depth: int = 6,
    ) -> None:
        super().__init__()
        c_in = int(in_channels)
        w0 = int(width)
        d = int(depth)
        if c_in <= 0:
            raise ValueError("in_channels must be > 0")
        if w0 < 8:
            raise ValueError("width must be >= 8")
        if d <= 0:
            raise ValueError("depth must be > 0")

        self.backbone = GatedPixelCNNBlindSpotBackbone(in_channels=c_in, width=w0, depth=d)
        self.fuse = nn.Sequential(
            nn.Conv2d(w0 * 4, w0, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(w0, c_in, kernel_size=3, padding=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        if x.ndim != 4:
            raise ValueError(f"Expected input shape (B, C, H, W), got {tuple(x.shape)}")

        feats: list[torch.Tensor] = []
        for k in (0, 1, 2, 3):
            xr = torch.rot90(x, k=k, dims=(-2, -1))
            fr = self.backbone(xr)
            f = torch.rot90(fr, k=-k, dims=(-2, -1))
            feats.append(f)

        f_cat = torch.cat(feats, dim=1)
        residual = self.fuse(f_cat)
        return x - residual


_VARIANTS: dict[str, dict] = {
    "gated_pixelcnn_bsn_tiny": {"width": 16, "depth": 3},
    "gated_pixelcnn_bsn_small": {"width": 24, "depth": 5},
    "gated_pixelcnn_bsn_base": {"width": 32, "depth": 7},
}


def build_gated_pixelcnn_bsn_denoiser(
    *,
    in_channels: int,
    variant: str = "gated_pixelcnn_bsn_small",
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(
            f"Unknown gated PixelCNN-BSN variant: {variant!r}. Supported: {sorted(_VARIANTS)}"
        )
    spec = _VARIANTS[name]
    return GatedPixelCNNBSN(
        in_channels=int(in_channels), width=int(spec["width"]), depth=int(spec["depth"])
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.rand(2, 1, 32, 32)
    noisy = (x + torch.randn_like(x) * 0.1).clamp(0.0, 1.0)
    m = build_gated_pixelcnn_bsn_denoiser(in_channels=1, variant="gated_pixelcnn_bsn_tiny")
    y = m(noisy)
    print("gated_pixelcnn_bsn_tiny", tuple(y.shape))
    loss = (y - x).pow(2).mean()
    loss.backward()
    print("ok")
