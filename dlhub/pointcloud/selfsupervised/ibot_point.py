import torch
from torch import nn

from dlhub.pointcloud.ops import farthest_point_sample, index_points, knn_query


class PatchEmbed(nn.Module):
    """Group points into patches and embed them into tokens.

    Returns:
        tokens: (B, S, D)
        centers: (B, S, 3)
        grouped_xyz: (B, S, K, 3) local coords (neighbor - center)
    """

    def __init__(
        self,
        *,
        in_channels: int,
        embed_dim: int,
        num_patches: int,
        group_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.embed_dim = int(embed_dim)
        self.num_patches = int(num_patches)
        self.group_size = int(group_size)

        d = int(embed_dim)
        self.point_embed = nn.Sequential(
            nn.Linear(int(in_channels), d),
            nn.ReLU(inplace=True),
        )
        self.group_mlp = nn.Sequential(
            nn.Linear(d + 3, d),
            nn.ReLU(inplace=True),
            nn.Dropout(p=float(dropout)),
            nn.Linear(d, d),
        )
        self.pos = nn.Sequential(
            nn.Linear(3, d),
            nn.ReLU(inplace=True),
            nn.Linear(d, d),
        )

    def forward(self, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if points.ndim != 3 or int(points.shape[-1]) != int(self.in_channels):
            raise ValueError(
                f"Expected points shape (B, N, C={self.in_channels}), got {tuple(points.shape)}"
            )

        xyz = points[..., :3].to(torch.float32)  # (B, N, 3)
        b, n, _ = xyz.shape
        s = int(self.num_patches)
        k = int(self.group_size)

        if n < k:
            raise ValueError(
                f"group_size must be <= num_points. Got group_size={k}, num_points={n}"
            )
        if s <= 0:
            raise ValueError("num_patches must be > 0")
        if s > n:
            s = int(n)

        feat = self.point_embed(points.to(torch.float32))  # (B, N, D)
        fps_idx = farthest_point_sample(xyz, s)  # (B, S)
        centers = index_points(xyz, fps_idx)  # (B, S, 3)

        idx = knn_query(k, xyz, centers)  # (B, S, K)
        grouped_xyz = index_points(xyz, idx) - centers.unsqueeze(2)  # (B, S, K, 3)
        grouped_feat = index_points(feat, idx)  # (B, S, K, D)

        x = torch.cat([grouped_feat, grouped_xyz], dim=-1)  # (B, S, K, D+3)
        x = self.group_mlp(x)  # (B, S, K, D)
        tokens = x.max(dim=2).values  # (B, S, D)
        tokens = tokens + self.pos(centers)
        return tokens, centers, grouped_xyz


def _gather_batch(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather x along dim=1 with per-batch indices.

    x: (B, S, ...)
    idx: (B, M)
    """

    if x.ndim < 2:
        raise ValueError("x must be at least 2D")
    if idx.ndim != 2:
        raise ValueError("idx must be (B, M)")
    b, _s = x.shape[0], x.shape[1]
    if idx.shape[0] != b:
        raise ValueError("Batch mismatch in _gather_batch")
    batch = torch.arange(b, device=x.device).unsqueeze(1)
    return x[batch, idx]


def _scatter_batch(x: torch.Tensor, idx: torch.Tensor, updates: torch.Tensor) -> torch.Tensor:
    """Scatter updates into x along dim=1 with per-batch indices."""

    if idx.ndim != 2:
        raise ValueError("idx must be (B, M)")
    b = x.shape[0]
    if idx.shape[0] != b or updates.shape[0] != b:
        raise ValueError("Batch mismatch in _scatter_batch")
    batch = torch.arange(b, device=x.device).unsqueeze(1)
    x = x.clone()
    x[batch, idx] = updates
    return x


class IBotPointPretrainer(nn.Module):
    """A compact PointMAE-style masked autoencoder (toy-first).

    This implementation:
    - groups points into S patches via FPS + kNN
    - masks a ratio of patches
    - encodes visible patch tokens with a Transformer
    - decodes full sequence (visible tokens + mask tokens)
    - reconstructs masked patches as K local point coordinates (relative to patch center)

    Forward returns:
    - pred: (B, M, K, 3) predicted local coords for masked patches
    - target: (B, M, K, 3) ground truth local coords for masked patches
    """

    def __init__(
        self,
        *,
        in_channels: int,
        embed_dim: int = 128,
        num_patches: int = 16,
        group_size: int = 16,
        encoder_depth: int = 4,
        encoder_heads: int = 4,
        decoder_dim: int = 96,
        decoder_depth: int = 2,
        decoder_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        d = int(embed_dim)
        dd = int(decoder_dim)
        if d <= 0 or dd <= 0:
            raise ValueError("embed_dim/decoder_dim must be > 0")

        self.patch = PatchEmbed(
            in_channels=int(in_channels),
            embed_dim=int(d),
            num_patches=int(num_patches),
            group_size=int(group_size),
            dropout=float(dropout),
        )

        enc_layer = nn.TransformerEncoderLayer(
            d_model=int(d),
            nhead=int(encoder_heads),
            dim_feedforward=int(d) * 4,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=int(encoder_depth))
        self.enc_norm = nn.LayerNorm(int(d))

        self.enc_to_dec = (
            nn.Linear(int(d), int(dd), bias=True) if int(dd) != int(d) else nn.Identity()
        )
        self.dec_pos = nn.Sequential(
            nn.Linear(3, int(dd)),
            nn.ReLU(inplace=True),
            nn.Linear(int(dd), int(dd)),
        )
        self.mask_token = nn.Parameter(torch.zeros(1, 1, int(dd)))

        dec_layer = nn.TransformerEncoderLayer(
            d_model=int(dd),
            nhead=int(decoder_heads),
            dim_feedforward=int(dd) * 4,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(dec_layer, num_layers=int(decoder_depth))
        self.dec_norm = nn.LayerNorm(int(dd))

        k = int(group_size)
        self.recon_head = nn.Sequential(
            nn.Linear(int(dd), int(dd)),
            nn.ReLU(inplace=True),
            nn.Linear(int(dd), k * 3),
        )

    def forward(self, points: torch.Tensor, *, mask_ratio: float = 0.6) -> dict[str, torch.Tensor]:
        mask_ratio = float(mask_ratio)
        if not (0.0 < mask_ratio < 1.0):
            raise ValueError("mask_ratio must be in (0, 1)")

        tokens, centers, grouped_xyz = self.patch(points)  # (B, S, D), (B, S, 3), (B, S, K, 3)
        b, s, _d = tokens.shape

        m = int(round(float(s) * mask_ratio))
        m = max(1, min(s - 1, m))

        # Sample per-sample mask indices by sorting random scores.
        scores = torch.rand((b, s), device=tokens.device, dtype=torch.float32)
        order = torch.argsort(scores, dim=1)  # (B, S)
        mask_idx = order[:, :m]  # (B, M)
        keep_idx = order[:, m:]  # (B, S-M)

        vis_tokens = _gather_batch(tokens, keep_idx)  # (B, S-M, D)
        vis = self.encoder(vis_tokens)
        vis = self.enc_norm(vis)

        vis_dec = self.enc_to_dec(vis)  # (B, S-M, Dd)
        full = self.mask_token.expand(b, s, -1)
        full = _scatter_batch(full, keep_idx, vis_dec)
        full = full + self.dec_pos(centers.to(full.dtype))

        dec = self.decoder(full)
        dec = self.dec_norm(dec)

        masked_tokens = _gather_batch(dec, mask_idx)  # (B, M, Dd)
        pred = self.recon_head(masked_tokens).view(b, m, self.patch.group_size, 3)
        target = _gather_batch(grouped_xyz, mask_idx)  # (B, M, K, 3)
        return {"pred": pred, "target": target}


_VARIANTS: dict[str, dict] = {
    "ibot_point_tiny": {
        "embed_dim": 96,
        "num_patches": 16,
        "group_size": 16,
        "enc_depth": 2,
        "enc_heads": 4,
        "dec_dim": 64,
        "dec_depth": 1,
        "dec_heads": 4,
    },
    "ibot_point_small": {
        "embed_dim": 128,
        "num_patches": 24,
        "group_size": 16,
        "enc_depth": 4,
        "enc_heads": 4,
        "dec_dim": 96,
        "dec_depth": 2,
        "dec_heads": 4,
    },
    "ibot_point_base": {
        "embed_dim": 192,
        "num_patches": 32,
        "group_size": 24,
        "enc_depth": 6,
        "enc_heads": 6,
        "dec_dim": 128,
        "dec_depth": 2,
        "dec_heads": 8,
    },
}


def build_ibot_point_pretrainer(
    *,
    in_channels: int,
    variant: str = "ibot_point_small",
    dropout: float = 0.0,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown PointMAE variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    return IBotPointPretrainer(
        in_channels=int(in_channels),
        embed_dim=int(spec["embed_dim"]),
        num_patches=int(spec["num_patches"]),
        group_size=int(spec["group_size"]),
        encoder_depth=int(spec["enc_depth"]),
        encoder_heads=int(spec["enc_heads"]),
        decoder_dim=int(spec["dec_dim"]),
        decoder_depth=int(spec["dec_depth"]),
        decoder_heads=int(spec["dec_heads"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    from dlhub.pointcloud.ops import chamfer_distance

    torch.manual_seed(0)
    pts = torch.randn(4, 128, 3)
    m = build_ibot_point_pretrainer(in_channels=3, variant="ibot_point_tiny", dropout=0.0)
    out = m(pts, mask_ratio=0.6)
    pred = out["pred"].reshape(-1, m.patch.group_size, 3)
    target = out["target"].reshape(-1, m.patch.group_size, 3)
    loss = chamfer_distance(pred, target)
    loss.backward()
    print("ok", float(loss.item()))


