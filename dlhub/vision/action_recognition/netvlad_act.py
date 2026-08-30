"""TimeSformer (space-time attention) - compact-first video action classifier.

Reference:
- "Is Space-Time Attention All You Need for Video Understanding?" (ICML 2021)

Compact interpretation:
- Patchify each frame with a Conv2d patch embed.
- Flatten all frame patches into a token sequence and run a TransformerEncoder.
- Use a CLS token for classification.
"""

import torch
from torch import nn

from dlhub.vision.backbones._blocks import scale_channels

from ._common import check_video_input


class NetvladActVideoClassifier(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        image_size: int,
        frames: int,
        patch_size: int,
        embed_dim: int,
        depth: int,
        heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        img = int(image_size)
        t = int(frames)
        p = int(patch_size)
        e = int(embed_dim)
        d = int(depth)
        h = int(heads)
        if img <= 0 or t <= 0:
            raise ValueError("image_size and frames must be > 0")
        if p <= 0:
            raise ValueError("patch_size must be > 0")
        if img % p != 0:
            raise ValueError(f"image_size ({img}) must be divisible by patch_size ({p})")
        if e <= 0 or d <= 0 or h <= 0:
            raise ValueError("embed_dim/depth/heads must be > 0")

        self.frames = t
        grid = img // p
        num_patches = grid * grid
        self.num_tokens = 1 + t * num_patches

        self.patch_embed = nn.Conv2d(int(in_channels), e, kernel_size=p, stride=p, bias=True)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, e))
        self.pos_embed = nn.Parameter(torch.zeros(1, int(self.num_tokens), e))

        layer = nn.TransformerEncoderLayer(
            d_model=e,
            nhead=h,
            dim_feedforward=max(64, e * 4),
            dropout=0.0,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=d, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(e)
        self.dropout = nn.Dropout(float(dropout))
        self.head = nn.Linear(e, int(num_classes))

        # A tiny init helps stable compact training loops.
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = check_video_input(x)  # (B,C,T,H,W)
        b, c, t, h, w = x.shape
        if int(t) != int(self.frames):
            raise ValueError(f"Expected T={self.frames} frames for this model, got T={t}")

        frames = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        tok = self.patch_embed(frames)  # (B*T, E, Gh, Gw)
        tok = tok.flatten(2).transpose(1, 2)  # (B*T, N, E)
        n = int(tok.shape[1])
        tok = tok.reshape(b, t * n, -1)  # (B, T*N, E)

        cls = self.cls_token.expand(b, -1, -1)  # (B,1,E)
        seq = torch.cat([cls, tok], dim=1)  # (B, 1+T*N, E)
        seq = seq + self.pos_embed[:, : seq.shape[1]]
        seq = self.encoder(seq)
        cls_out = self.norm(seq[:, 0])
        cls_out = self.dropout(cls_out)
        return self.head(cls_out)


_VARIANTS: dict[str, dict] = {
    "netvlad_act_tiny": {"patch": 8, "embed": 96, "depth": 2, "heads": 4},
    "netvlad_act_small": {"patch": 8, "embed": 128, "depth": 3, "heads": 4},
    "netvlad_act_base": {"patch": 4, "embed": 160, "depth": 4, "heads": 8},
}


def build_netvlad_act_video_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "netvlad_act_small",
    image_size: int = 64,
    frames: int = 8,
    width_mult: float = 1.0,
    dropout: float = 0.1,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(
            f"Unknown TimeSformer variant: {variant!r}. Supported: {sorted(_VARIANTS)}"
        )
    spec = _VARIANTS[name]
    embed = scale_channels(int(spec["embed"]), float(width_mult), min_ch=32, divisor=8)
    return NetvladActVideoClassifier(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        image_size=int(image_size),
        frames=int(frames),
        patch_size=int(spec["patch"]),
        embed_dim=int(embed),
        depth=int(spec["depth"]),
        heads=int(spec["heads"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 8, 64, 64)
    m = build_netvlad_act_video_classifier(
        in_channels=3, num_classes=6, variant="netvlad_act_tiny", width_mult=0.5, dropout=0.0
    )
    y = m(x)
    print("netvlad_act_tiny", tuple(y.shape))
    loss = y.mean()
    loss.backward()
    print("ok")
