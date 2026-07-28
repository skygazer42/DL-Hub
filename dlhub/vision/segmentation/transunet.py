import torch
import torch.nn.functional as F
from torch import nn

from dlhub.vision.backbones._blocks import scale_channels
from dlhub.vision.segmentation._common import check_nchw


class _DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        c_in = int(in_ch)
        c_out = int(out_ch)
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = _DoubleConv(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class _Up(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(int(in_ch), int(out_ch), kernel_size=2, stride=2, bias=True)
        self.conv = _DoubleConv(int(out_ch) * 2, int(out_ch))

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class TransUNet(nn.Module):
    """TransUNet semantic segmentation (compact-first).

    U-Net encoder-decoder with a Transformer encoder at the bottleneck.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        base_channels: int = 32,
        levels: int = 4,
        transformer_depth: int = 2,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        nc = int(num_classes)
        if nc <= 0:
            raise ValueError("num_classes must be > 0")
        base = int(base_channels)
        lv = int(levels)
        if lv < 2:
            raise ValueError("levels must be >= 2")

        self.inc = _DoubleConv(int(in_channels), base)

        downs: list[nn.Module] = []
        ch = base
        for _ in range(lv - 1):
            downs.append(_Down(ch, ch * 2))
            ch *= 2
        self.downs = nn.ModuleList(downs)

        # Transformer at bottleneck, operating on (H*W) tokens with dim=ch.
        d = int(transformer_depth)
        h = int(num_heads)
        if d <= 0:
            raise ValueError("transformer_depth must be > 0")
        if h <= 0 or ch % h != 0:
            raise ValueError("num_heads must be > 0 and divide bottleneck channels")
        ff = int(round(ch * float(mlp_ratio)))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=ch,
            nhead=h,
            dim_feedforward=ff,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=d)

        ups: list[nn.Module] = []
        for _ in range(lv - 1):
            ups.append(_Up(ch, ch // 2))
            ch //= 2
        self.ups = nn.ModuleList(ups)

        self.outc = nn.Conv2d(base, nc, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = check_nchw(x)
        inp_hw = x.shape[-2:]

        skips: list[torch.Tensor] = []
        x = self.inc(x)
        skips.append(x)
        for down in self.downs:
            x = down(x)
            skips.append(x)

        b, c, h, w = x.shape
        tok = x.permute(0, 2, 3, 1).reshape(b, h * w, c)
        tok = self.transformer(tok)
        x = tok.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()

        for up, skip in zip(self.ups, reversed(skips[:-1]), strict=True):
            x = up(x, skip)

        logits = self.outc(x)
        return F.interpolate(logits, size=inp_hw, mode="nearest")


_VARIANTS: dict[str, dict] = {
    "transunet_tiny": {"base_channels": 16, "levels": 3, "tdepth": 1, "heads": 4},
    "transunet_small": {"base_channels": 24, "levels": 4, "tdepth": 2, "heads": 4},
    "transunet_base": {"base_channels": 32, "levels": 4, "tdepth": 3, "heads": 8},
}


def build_transunet_segmenter(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "transunet_small",
    width_mult: float = 1.0,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown TransUNet variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    base = scale_channels(int(spec["base_channels"]), float(width_mult), min_ch=16, divisor=8)
    heads = int(spec["heads"])
    bottleneck = int(base) * (2 ** (int(spec["levels"]) - 1))
    while heads > 1 and bottleneck % heads != 0:
        heads -= 1
    return TransUNet(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        base_channels=int(base),
        levels=int(spec["levels"]),
        transformer_depth=int(spec["tdepth"]),
        num_heads=int(heads),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_transunet_segmenter(
        in_channels=3, num_classes=4, variant="transunet_tiny", width_mult=0.5
    )
    y = m(x)
    print("transunet_tiny", tuple(y.shape))
    loss = y.mean()
    loss.backward()
    print("ok")
