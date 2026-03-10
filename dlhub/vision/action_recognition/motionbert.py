"""MotionBERT (masked motion modeling) - toy-first skeleton action classifier.

Reference (idea):
- "MotionBERT: Unified Pretraining for Human Motion Analysis" (arXiv/CVPR-era 2022)

Toy interpretation for action recognition:
- Tokenize (time, joint) into a sequence.
- Optionally mask a random subset of tokens during training (masked motion modeling flavor).
- TransformerEncoder + CLS token classification.
"""

import torch
from torch import nn

from dlhub.vision.backbones._blocks import scale_channels

from ._common import check_skeleton_input


class MotionBERTSkeletonClassifier(nn.Module):
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
        mask_ratio: float,
        dropout: float,
    ) -> None:
        super().__init__()
        c_in = int(in_channels)
        v = int(num_joints)
        t = int(seq_len)
        e = int(embed_dim)
        d = int(depth)
        h = int(heads)
        mr = float(mask_ratio)
        if v <= 0 or t <= 0:
            raise ValueError("num_joints and seq_len must be > 0")
        if e <= 0 or d <= 0 or h <= 0:
            raise ValueError("embed_dim/depth/heads must be > 0")
        if not (0.0 <= mr < 1.0):
            raise ValueError("mask_ratio must be in [0, 1)")

        self.num_joints = v
        self.seq_len = t
        self.mask_ratio = mr

        n = int(t * v)
        self.proj = nn.Linear(c_in, e)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, e))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, e))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + n, e))

        layer = nn.TransformerEncoderLayer(
            d_model=e,
            nhead=h,
            dim_feedforward=max(64, e * 4),
            dropout=0.0,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=d)
        self.norm = nn.LayerNorm(e)
        self.dropout = nn.Dropout(float(dropout))
        self.head = nn.Linear(e, int(num_classes))

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = check_skeleton_input(x)  # (B,C,T,V)
        b, c, t, v = x.shape
        if int(t) != int(self.seq_len) or int(v) != int(self.num_joints):
            raise ValueError(
                f"Expected (T,V)=({self.seq_len},{self.num_joints}), got (T,V)=({t},{v})"
            )

        tokens = x.permute(0, 2, 3, 1).reshape(b, t * v, c)  # (B, N, C)
        tokens = self.proj(tokens)  # (B, N, E)

        if self.training and float(self.mask_ratio) > 0.0:
            n = int(tokens.shape[1])
            mask = torch.rand(b, n, device=tokens.device) < float(self.mask_ratio)
            mtok = self.mask_token.expand(b, n, -1)
            tokens = torch.where(mask.unsqueeze(-1), mtok, tokens)

        cls = self.cls_token.expand(b, -1, -1)
        seq = torch.cat([cls, tokens], dim=1)
        seq = seq + self.pos_embed[:, : seq.shape[1]]
        seq = self.encoder(seq)
        out = self.norm(seq[:, 0])
        out = self.dropout(out)
        return self.head(out)


_VARIANTS: dict[str, dict] = {
    "motionbert_tiny": {"embed": 96, "depth": 2, "heads": 4, "mask_ratio": 0.0},
    "motionbert_small": {"embed": 128, "depth": 3, "heads": 4, "mask_ratio": 0.15},
    "motionbert_base": {"embed": 160, "depth": 4, "heads": 8, "mask_ratio": 0.25},
}


def build_motionbert_skeleton_classifier(
    *,
    in_channels: int,
    num_classes: int,
    num_joints: int = 17,
    seq_len: int = 32,
    variant: str = "motionbert_small",
    width_mult: float = 1.0,
    dropout: float = 0.1,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown MotionBERT variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    embed = scale_channels(int(spec["embed"]), float(width_mult), min_ch=32, divisor=8)
    return MotionBERTSkeletonClassifier(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        num_joints=int(num_joints),
        seq_len=int(seq_len),
        embed_dim=int(embed),
        depth=int(spec["depth"]),
        heads=int(spec["heads"]),
        mask_ratio=float(spec["mask_ratio"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 32, 17)
    m = build_motionbert_skeleton_classifier(
        in_channels=3,
        num_classes=6,
        num_joints=17,
        seq_len=32,
        variant="motionbert_tiny",
        width_mult=0.5,
        dropout=0.0,
    )
    y = m(x)
    print("motionbert_tiny", tuple(y.shape))
    loss = y.mean()
    loss.backward()
    print("ok")
