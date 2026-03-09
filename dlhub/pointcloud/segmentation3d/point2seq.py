
import torch
from torch import nn

from ._common import check_points, split_xyz_features


_VARIANTS: dict[str, dict[str, object]] = {
    "point2seq_tiny": {"width": 64, "layers": 1},
    "point2seq_small": {"width": 96, "layers": 2},
    "point2seq_base": {"width": 128, "layers": 3},
}


class Point2SeqSeg(nn.Module):
    """Point2Seq semantic segmentation (toy): sort points, run GRU, then unsort."""

    def __init__(self, *, in_channels: int, num_classes: int, width: int, layers: int, dropout: float = 0.0) -> None:
        super().__init__()
        w = int(width)
        self.embed = nn.Sequential(nn.Linear(int(in_channels), w), nn.ReLU(inplace=True))
        self.rnn = nn.GRU(w, w, num_layers=int(layers), batch_first=True, bidirectional=True, dropout=float(dropout))
        self.proj = nn.Sequential(nn.Linear(w * 2, w), nn.ReLU(inplace=True))
        self.cls = nn.Sequential(nn.Linear(w, w), nn.ReLU(inplace=True), nn.Linear(w, int(num_classes)))

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        check_points(points)
        xyz, feats = split_xyz_features(points)
        x = xyz if feats is None else torch.cat([xyz, feats], dim=-1)
        h = self.embed(x.to(torch.float32))

        # Sort by x coordinate for a deterministic "sequence".
        order = xyz[..., 0].argsort(dim=1)  # (B,N)
        b = points.shape[0]
        batch = torch.arange(b, device=points.device).unsqueeze(-1)
        hs = h[batch, order]  # (B,N,W)
        ys, _ = self.rnn(hs)
        ys = self.proj(ys)

        # Unsort back.
        inv = order.argsort(dim=1)
        y = ys[batch, inv]
        return self.cls(y)


def build_point2seq_segmenter3d(
    in_channels: int,
    num_classes: int,
    *,
    variant: str = "point2seq_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    cfg = _VARIANTS[str(variant)]
    width = int(int(cfg["width"]) * float(width_mult))
    return Point2SeqSeg(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        width=width,
        layers=int(cfg["layers"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    model = build_point2seq_segmenter3d(in_channels=3, num_classes=6, variant="point2seq_tiny")
    x = torch.randn(2, 128, 3)
    y = model(x)
    y.mean().backward()
    print("logits:", tuple(y.shape))

