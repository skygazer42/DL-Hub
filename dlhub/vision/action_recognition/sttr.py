"""ST-Transformer (factorized attention) - toy-first skeleton action classifier.

Reference (representative idea):
- Spatio-temporal transformer variants for skeleton action recognition (around 2021).

Toy interpretation:
- Project joint coordinates into tokens (T * V tokens).
- Factorize attention:
  1) spatial attention over joints for each time step
  2) temporal attention over time for each joint
- Global average pool over (T, V) then classify.
"""

import torch
from torch import nn

from dlhub.vision.backbones._blocks import scale_channels

from ._common import check_skeleton_input


class STTRSkeletonClassifier(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        num_joints: int,
        seq_len: int,
        embed_dim: int,
        depth: int,
        heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        c_in = int(in_channels)
        v = int(num_joints)
        t = int(seq_len)
        e = int(embed_dim)
        d = int(depth)
        h = int(heads)
        if v <= 0 or t <= 0:
            raise ValueError("num_joints and seq_len must be > 0")
        if e <= 0 or d <= 0 or h <= 0:
            raise ValueError("embed_dim/depth/heads must be > 0")

        self.num_joints = v
        self.seq_len = t

        self.proj = nn.Linear(c_in, e)
        self.time_embed = nn.Parameter(torch.zeros(1, t, 1, e))
        self.joint_embed = nn.Parameter(torch.zeros(1, 1, v, e))

        spatial_layer = nn.TransformerEncoderLayer(
            d_model=e,
            nhead=h,
            dim_feedforward=max(64, e * 4),
            dropout=0.0,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=e,
            nhead=h,
            dim_feedforward=max(64, e * 4),
            dropout=0.0,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )

        self.spatial = nn.TransformerEncoder(spatial_layer, num_layers=d)
        self.temporal = nn.TransformerEncoder(temporal_layer, num_layers=d)

        self.norm = nn.LayerNorm(e)
        self.dropout = nn.Dropout(float(dropout))
        self.head = nn.Linear(e, int(num_classes))

        nn.init.trunc_normal_(self.time_embed, std=0.02)
        nn.init.trunc_normal_(self.joint_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = check_skeleton_input(x)  # (B,C,T,V)
        b, c, t, v = x.shape
        if int(t) != int(self.seq_len) or int(v) != int(self.num_joints):
            raise ValueError(
                f"Expected (T,V)=({self.seq_len},{self.num_joints}), got (T,V)=({t},{v})"
            )

        # (B,C,T,V) -> (B,T,V,C) -> (B,T,V,E)
        tok = x.permute(0, 2, 3, 1).contiguous()
        tok = self.proj(tok) + self.time_embed + self.joint_embed

        # Spatial attention per time step: (B*T, V, E)
        tok_s = tok.view(b * t, v, -1)
        tok_s = self.spatial(tok_s)
        tok = tok_s.view(b, t, v, -1)

        # Temporal attention per joint: (B*V, T, E)
        tok_t = tok.permute(0, 2, 1, 3).contiguous().view(b * v, t, -1)
        tok_t = self.temporal(tok_t)
        tok = tok_t.view(b, v, t, -1).permute(0, 2, 1, 3).contiguous()

        pooled = self.norm(tok).mean(dim=(1, 2))
        pooled = self.dropout(pooled)
        return self.head(pooled)


_VARIANTS: dict[str, dict] = {
    "sttr_tiny": {"embed": 96, "depth": 1, "heads": 4},
    "sttr_small": {"embed": 128, "depth": 2, "heads": 4},
    "sttr_base": {"embed": 160, "depth": 3, "heads": 8},
}


def build_sttr_skeleton_classifier(
    *,
    in_channels: int,
    num_classes: int,
    num_joints: int = 17,
    seq_len: int = 32,
    variant: str = "sttr_small",
    width_mult: float = 1.0,
    dropout: float = 0.1,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown ST-TR variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    embed = scale_channels(int(spec["embed"]), float(width_mult), min_ch=32, divisor=8)
    return STTRSkeletonClassifier(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        num_joints=int(num_joints),
        seq_len=int(seq_len),
        embed_dim=int(embed),
        depth=int(spec["depth"]),
        heads=int(spec["heads"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 32, 17)
    m = build_sttr_skeleton_classifier(
        in_channels=3,
        num_classes=6,
        num_joints=17,
        seq_len=32,
        variant="sttr_tiny",
        width_mult=0.5,
        dropout=0.0,
    )
    y = m(x)
    print("sttr_tiny", tuple(y.shape))
    loss = y.mean()
    loss.backward()
    print("ok")
