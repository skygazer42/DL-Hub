import torch
import torch.nn.functional as F
from torch import nn

from dlhub.vision.backbones._blocks import ConvBNAct, scale_channels
from dlhub.vision.segmentation._common import check_nchw


class FANet(nn.Module):
    """FANet-style semantic segmentation (toy-first).

    This compact version uses a single patch embedding stage + Transformer encoder,
    then a lightweight conv head to predict logits.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        embed_dim: int = 96,
        depth: int = 3,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        k = int(num_classes)
        if k <= 0:
            raise ValueError("num_classes must be > 0")
        dim = int(embed_dim)
        if dim <= 0:
            raise ValueError("embed_dim must be > 0")
        h = int(num_heads)
        if h <= 0 or dim % h != 0:
            raise ValueError("num_heads must be > 0 and divide embed_dim")
        d = int(depth)
        if d <= 0:
            raise ValueError("depth must be > 0")

        self.patch = nn.Sequential(
            ConvBNAct(int(in_channels), dim, kernel_size=7, stride=4, padding=3, act="relu"),
        )

        ff = int(round(dim * float(mlp_ratio)))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=h,
            dim_feedforward=ff,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=d)

        self.head = nn.Sequential(
            ConvBNAct(dim, dim, kernel_size=3, stride=1, act="relu"),
            nn.Conv2d(dim, k, kernel_size=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = check_nchw(x)
        inp_hw = x.shape[-2:]

        feat = self.patch(x)  # (B, D, H/4, W/4)
        b, d, h, w = feat.shape
        tok = feat.permute(0, 2, 3, 1).reshape(b, h * w, d)  # (B, N, D)
        tok = self.encoder(tok)
        feat = tok.view(b, h, w, d).permute(0, 3, 1, 2).contiguous()

        logits4 = self.head(feat)
        return F.interpolate(logits4, size=inp_hw, mode="nearest")


_VARIANTS: dict[str, dict] = {
    "fanet_tiny": {"embed_dim": 64, "depth": 2, "num_heads": 4, "mlp_ratio": 4.0},
    "fanet_small": {"embed_dim": 96, "depth": 3, "num_heads": 4, "mlp_ratio": 4.0},
    "fanet_base": {"embed_dim": 128, "depth": 4, "num_heads": 8, "mlp_ratio": 4.0},
}


def build_fanet_segmenter(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "fanet_small",
    width_mult: float = 1.0,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown FANet variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    dim = scale_channels(int(spec["embed_dim"]), float(width_mult), min_ch=32, divisor=8)
    heads = int(spec["num_heads"])
    # Keep heads dividing dim.
    while heads > 1 and dim % heads != 0:
        heads -= 1
    return FANet(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        embed_dim=int(dim),
        depth=int(spec["depth"]),
        num_heads=int(heads),
        mlp_ratio=float(spec["mlp_ratio"]),
        dropout=0.0,
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_fanet_segmenter(in_channels=3, num_classes=4, variant="fanet_tiny", width_mult=0.5)
    y = m(x)
    print("fanet_tiny", tuple(y.shape))
    loss = y.mean()
    loss.backward()
    print("ok")
