import torch
import torch.nn.functional as F
from torch import nn

from ._common import ConvBNAct, SegmentationDecoder, TinyLaneEncoder, scaled_channels


class ENetSADLaneDetector(nn.Module):
    """A tiny ENet-SAD style detector with main and auxiliary segmentation heads."""

    def __init__(
        self,
        *,
        in_channels: int,
        stem_channels: int,
        hidden_channels: int,
        depth: int,
        aux_channels: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.encoder = TinyLaneEncoder(
            in_channels=int(in_channels),
            stem_channels=int(stem_channels),
            hidden_channels=int(hidden_channels),
            depth=int(depth),
            dropout=float(dropout),
        )
        self.decoder = SegmentationDecoder(
            low_channels=int(hidden_channels),
            high_channels=int(hidden_channels),
            out_channels=int(hidden_channels),
            dropout=float(dropout),
        )
        self.aux_proj = nn.Sequential(
            ConvBNAct(int(hidden_channels), int(aux_channels)),
            nn.Conv2d(int(aux_channels), 1, kernel_size=1),
        )
        self.main_head = nn.Sequential(
            ConvBNAct(int(hidden_channels), int(hidden_channels)),
            nn.Conv2d(int(hidden_channels), 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = x.to(torch.float32)
        if x.ndim != 4:
            raise ValueError(f"Expected input shape (B, C, H, W), got {tuple(x.shape)}")

        low, high = self.encoder(x)
        feats = self.decoder(low, high, output_size=tuple(x.shape[-2:]))
        aux = self.aux_proj(high)
        aux = F.interpolate(aux, size=tuple(x.shape[-2:]), mode="bilinear", align_corners=False)
        return {"binary_logits": self.main_head(feats), "aux_logits": aux}


_VARIANTS: dict[str, dict[str, int | float]] = {
    "enet_sad_tiny": {"stem": 16, "hidden": 32, "depth": 1, "aux": 16, "dropout": 0.0},
    "enet_sad_small": {"stem": 24, "hidden": 48, "depth": 2, "aux": 24, "dropout": 0.0},
    "enet_sad_base": {"stem": 32, "hidden": 64, "depth": 3, "aux": 32, "dropout": 0.1},
}


def build_enet_sad_lane_detector(
    *,
    in_channels: int,
    num_lanes: int,
    image_size: int = 64,
    num_points: int = 16,
    num_rows: int = 16,
    grid_size: int = 32,
    num_anchors: int = 24,
    num_queries: int = 6,
    variant: str = "enet_sad_small",
    width_mult: float = 1.0,
    dropout: float | None = None,
) -> nn.Module:
    del num_lanes, image_size, num_points, num_rows, grid_size, num_anchors, num_queries
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(
            f"Unknown ENet-SAD variant: {variant!r}. Supported: {sorted(_VARIANTS)}"
        )

    spec = _VARIANTS[name]
    stem = scaled_channels(int(spec["stem"]), float(width_mult))
    hidden = scaled_channels(int(spec["hidden"]), float(width_mult))
    aux = scaled_channels(int(spec["aux"]), float(width_mult))
    p = float(spec["dropout"]) if dropout is None else float(dropout)
    return ENetSADLaneDetector(
        in_channels=int(in_channels),
        stem_channels=int(stem),
        hidden_channels=int(hidden),
        depth=int(spec["depth"]),
        aux_channels=int(aux),
        dropout=float(p),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_enet_sad_lane_detector(
        in_channels=3,
        num_lanes=4,
        variant="enet_sad_tiny",
        width_mult=1.0,
    )
    out = m(x)
    print("enet_sad_tiny", {k: tuple(v.shape) for k, v in out.items()})
    loss = sum(v.mean() for v in out.values())
    loss.backward()
    print("ok")
