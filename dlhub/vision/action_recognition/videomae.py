"""VideoMAE (tubelet ViT) - toy-first video action classifier.

Reference:
- "VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training"
  (NeurIPS 2022)

Toy interpretation for classification:
- Tubelet embedding via Conv3d (tubelet, patch, patch) stride.
- TransformerEncoder over flattened spatiotemporal tokens + a CLS token.
- No masked autoencoding objective here; this is a structural toy for quick comparisons.
"""

import torch
from torch import nn

from dlhub.vision.backbones._blocks import scale_channels

from ._common import check_video_input


class VideoMAEVideoClassifier(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        image_size: int,
        frames: int,
        patch_size: int,
        tubelet: int,
        embed_dim: int,
        depth: int,
        heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        img = int(image_size)
        t = int(frames)
        p = int(patch_size)
        tb = int(tubelet)
        e = int(embed_dim)
        d = int(depth)
        h = int(heads)
        if img <= 0 or t <= 0:
            raise ValueError("image_size and frames must be > 0")
        if p <= 0 or img % p != 0:
            raise ValueError("patch_size must be > 0 and divide image_size")
        if tb <= 0 or t % tb != 0:
            raise ValueError("tubelet must be > 0 and divide frames")
        if e <= 0 or d <= 0 or h <= 0:
            raise ValueError("embed_dim/depth/heads must be > 0")

        self.frames = t
        self.image_size = img
        self.patch_size = p
        self.tubelet = tb

        t_grid = t // tb
        h_grid = img // p
        w_grid = img // p
        num_tokens = int(t_grid * h_grid * w_grid)
        self.num_tokens = num_tokens

        self.tubelet_embed = nn.Conv3d(
            int(in_channels),
            e,
            kernel_size=(tb, p, p),
            stride=(tb, p, p),
            padding=0,
            bias=True,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, e))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + num_tokens, e))

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = check_video_input(x)
        b, c, t, h, w = x.shape
        if int(t) != int(self.frames):
            raise ValueError(f"Expected T={self.frames} frames for this model, got T={t}")
        if int(h) != int(self.image_size) or int(w) != int(self.image_size):
            raise ValueError(
                f"Expected H=W=image_size={self.image_size} for this model, got (H,W)=({h},{w})"
            )

        tok = self.tubelet_embed(x)  # (B, E, T', H', W')
        tok = tok.flatten(2).transpose(1, 2).contiguous()  # (B, N, E)
        if int(tok.shape[1]) != int(self.num_tokens):
            raise ValueError(
                f"Unexpected token count: got N={tok.shape[1]}, expected {self.num_tokens}"
            )

        cls = self.cls_token.expand(b, -1, -1)
        seq = torch.cat([cls, tok], dim=1)
        seq = seq + self.pos_embed[:, : seq.shape[1]]
        seq = self.encoder(seq)
        out = self.norm(seq[:, 0])
        out = self.dropout(out)
        return self.head(out)


_VARIANTS: dict[str, dict] = {
    "videomae_tiny": {"patch": 8, "tubelet": 2, "embed": 128, "depth": 2, "heads": 4},
    "videomae_small": {"patch": 8, "tubelet": 2, "embed": 160, "depth": 3, "heads": 4},
    "videomae_base": {"patch": 4, "tubelet": 2, "embed": 192, "depth": 4, "heads": 8},
}


def build_videomae_video_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "videomae_small",
    image_size: int = 64,
    frames: int = 8,
    width_mult: float = 1.0,
    dropout: float = 0.1,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown VideoMAE variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    embed = scale_channels(int(spec["embed"]), float(width_mult), min_ch=32, divisor=8)
    return VideoMAEVideoClassifier(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        image_size=int(image_size),
        frames=int(frames),
        patch_size=int(spec["patch"]),
        tubelet=int(spec["tubelet"]),
        embed_dim=int(embed),
        depth=int(spec["depth"]),
        heads=int(spec["heads"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 8, 64, 64)
    m = build_videomae_video_classifier(
        in_channels=3,
        num_classes=6,
        variant="videomae_tiny",
        image_size=64,
        frames=8,
        width_mult=0.5,
        dropout=0.0,
    )
    y = m(x)
    print("videomae_tiny", tuple(y.shape))
    loss = y.mean()
    loss.backward()
    print("ok")
