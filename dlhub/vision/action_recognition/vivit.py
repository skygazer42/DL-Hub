"""ViViT (factorized video transformer) - toy-first video action classifier.

Reference:
- "ViViT: A Video Vision Transformer" (ICCV 2021)

Toy interpretation (factorized encoder):
- Patchify each frame with a Conv2d patch embed.
- Spatial encoder: self-attention over patches within each frame (shared across frames).
- Temporal encoder: self-attention over the sequence of per-frame embeddings.
"""

import torch
from torch import nn

from dlhub.vision.backbones._blocks import scale_channels

from ._common import check_video_input


class ViViTVideoClassifier(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        image_size: int,
        frames: int,
        patch_size: int,
        embed_dim: int,
        spatial_depth: int,
        temporal_depth: int,
        heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        img = int(image_size)
        t = int(frames)
        p = int(patch_size)
        e = int(embed_dim)
        sd = int(spatial_depth)
        td = int(temporal_depth)
        h = int(heads)
        if img <= 0 or t <= 0:
            raise ValueError("image_size and frames must be > 0")
        if p <= 0 or img % p != 0:
            raise ValueError("patch_size must be > 0 and divide image_size")
        if e <= 0 or sd <= 0 or td <= 0 or h <= 0:
            raise ValueError("embed_dim/depth/heads must be > 0")

        grid = img // p
        num_patches = grid * grid
        self.frames = t
        self.num_patches = int(num_patches)

        self.patch_embed = nn.Conv2d(int(in_channels), e, kernel_size=p, stride=p, bias=True)
        self.pos_spatial = nn.Parameter(torch.zeros(1, int(num_patches), e))

        spatial_layer = nn.TransformerEncoderLayer(
            d_model=e,
            nhead=h,
            dim_feedforward=max(64, e * 4),
            dropout=0.0,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.spatial = nn.TransformerEncoder(spatial_layer, num_layers=sd)

        # Temporal stage works on per-frame embeddings.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, e))
        self.pos_temporal = nn.Parameter(torch.zeros(1, 1 + t, e))

        temporal_layer = nn.TransformerEncoderLayer(
            d_model=e,
            nhead=h,
            dim_feedforward=max(64, e * 4),
            dropout=0.0,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(temporal_layer, num_layers=td)

        self.norm = nn.LayerNorm(e)
        self.dropout = nn.Dropout(float(dropout))
        self.head = nn.Linear(e, int(num_classes))

        nn.init.trunc_normal_(self.pos_spatial, std=0.02)
        nn.init.trunc_normal_(self.pos_temporal, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = check_video_input(x)
        b, c, t, h, w = x.shape
        if int(t) != int(self.frames):
            raise ValueError(f"Expected T={self.frames} frames for this model, got T={t}")

        # --- Spatial encoding per-frame: (B*T, N, E)
        frames = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        tok = self.patch_embed(frames).flatten(2).transpose(1, 2)  # (BT, N, E)
        if int(tok.shape[1]) != int(self.num_patches):
            raise ValueError(f"Unexpected patch token count: got N={tok.shape[1]}, expected {self.num_patches}")
        tok = tok + self.pos_spatial
        tok = self.spatial(tok)

        # Pool tokens into a per-frame embedding: (B, T, E)
        frame_emb = tok.mean(dim=1).view(b, t, -1)

        # --- Temporal encoding: (B, 1+T, E)
        cls = self.cls_token.expand(b, -1, -1)
        seq = torch.cat([cls, frame_emb], dim=1)
        seq = seq + self.pos_temporal[:, : seq.shape[1]]
        seq = self.temporal(seq)
        out = self.norm(seq[:, 0])
        out = self.dropout(out)
        return self.head(out)


_VARIANTS: dict[str, dict] = {
    "vivit_tiny": {"patch": 8, "embed": 96, "sdepth": 1, "tdepth": 2, "heads": 4},
    "vivit_small": {"patch": 8, "embed": 128, "sdepth": 2, "tdepth": 2, "heads": 4},
    "vivit_base": {"patch": 4, "embed": 160, "sdepth": 2, "tdepth": 3, "heads": 8},
}


def build_vivit_video_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "vivit_small",
    image_size: int = 64,
    frames: int = 8,
    width_mult: float = 1.0,
    dropout: float = 0.1,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown ViViT variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    embed = scale_channels(int(spec["embed"]), float(width_mult), min_ch=32, divisor=8)
    return ViViTVideoClassifier(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        image_size=int(image_size),
        frames=int(frames),
        patch_size=int(spec["patch"]),
        embed_dim=int(embed),
        spatial_depth=int(spec["sdepth"]),
        temporal_depth=int(spec["tdepth"]),
        heads=int(spec["heads"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 8, 64, 64)
    m = build_vivit_video_classifier(in_channels=3, num_classes=6, variant="vivit_tiny", image_size=64, frames=8, width_mult=0.5, dropout=0.0)
    y = m(x)
    print("vivit_tiny", tuple(y.shape))
    loss = y.mean()
    loss.backward()
    print("ok")

