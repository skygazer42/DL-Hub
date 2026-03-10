import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


def check_points(points: torch.Tensor) -> None:
    if not isinstance(points, torch.Tensor):
        raise TypeError(f"points must be a torch.Tensor, got {type(points).__name__}")
    if points.ndim != 3:
        raise ValueError(f"points must have shape (B, N, C), got {tuple(points.shape)}")
    if points.shape[-1] < 3:
        raise ValueError(f"points last dim must be >=3 (xyz), got C={points.shape[-1]}")


def split_xyz_features(points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
    xyz = points[..., :3]
    feats = points[..., 3:] if points.shape[-1] > 3 else None
    return xyz, feats


def normalize_xyz(
    xyz: torch.Tensor,
    *,
    center: bool = True,
    scale: bool = True,
    eps: float = 1e-6,
) -> torch.Tensor:
    if center:
        xyz = xyz - xyz.mean(dim=1, keepdim=True)
    if scale:
        std = xyz.std(dim=1, keepdim=True).clamp_min(eps)
        xyz = xyz / std
    return xyz


def mlp(in_dim: int, hidden: list[int], out_dim: int, *, dropout: float = 0.0) -> nn.Sequential:
    layers: list[nn.Module] = []
    d = int(in_dim)
    for h in hidden:
        h = int(h)
        layers.extend([nn.Linear(d, h), nn.ReLU(inplace=True)])
        if dropout and dropout > 0:
            layers.append(nn.Dropout(float(dropout)))
        d = h
    layers.append(nn.Linear(d, int(out_dim)))
    return nn.Sequential(*layers)


class PointNetEncoder(nn.Module):
    """Tiny point encoder (PointNet-ish)."""

    def __init__(self, in_channels: int, *, width: int, dropout: float = 0.0) -> None:
        super().__init__()
        in_channels = int(in_channels)
        width = int(width)

        self.fc1 = nn.Linear(in_channels, width)
        self.fc2 = nn.Linear(width, width)
        self.fc3 = nn.Linear(width, width)
        self.drop = nn.Dropout(float(dropout)) if dropout and dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, C)
        x = F.relu(self.fc1(x), inplace=True)
        x = self.drop(x)
        x = F.relu(self.fc2(x), inplace=True)
        x = self.drop(x)
        x = F.relu(self.fc3(x), inplace=True)
        return x  # (B, N, D)


def knn_indices(x: torch.Tensor, k: int) -> torch.Tensor:
    """Return kNN indices for each point (brute force; OK for toy N).

    x: (B, N, D)
    returns: (B, N, k) long
    """

    k = int(k)
    if k <= 0:
        raise ValueError("k must be > 0")
    if x.ndim != 3:
        raise ValueError(f"Expected (B,N,D), got {tuple(x.shape)}")

    # Pairwise distances: (B,N,N)
    # ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a·b
    xx = (x**2).sum(dim=-1, keepdim=True)  # (B,N,1)
    dist = xx + xx.transpose(1, 2) - 2 * (x @ x.transpose(1, 2))
    dist = dist.clamp_min(0.0)

    # Exclude self by setting diagonal large
    n = x.shape[1]
    eye = torch.eye(n, device=x.device, dtype=torch.bool).unsqueeze(0)
    dist = dist.masked_fill(eye, float("inf"))

    _, idx = dist.topk(k, dim=-1, largest=False)
    return idx


class EdgeConv(nn.Module):
    """DGCNN-style EdgeConv block (toy)."""

    def __init__(
        self, in_channels: int, out_channels: int, *, k: int = 16, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.k = int(k)
        self.mlp = nn.Sequential(
            nn.Linear(int(in_channels) * 2, int(out_channels)),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)) if dropout and dropout > 0 else nn.Identity(),
            nn.Linear(int(out_channels), int(out_channels)),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,N,C)
        idx = knn_indices(x, self.k)  # (B,N,k)
        b, n, c = x.shape
        idx.shape[-1]

        from dlhub.pointcloud.ops import index_points

        neighbors = index_points(x, idx)  # (B,N,k,C)
        x_i = x.unsqueeze(2).expand_as(neighbors)
        edge = torch.cat([x_i, neighbors - x_i], dim=-1)  # (B,N,k,2C)
        edge = self.mlp(edge)  # (B,N,k,D)
        return edge.max(dim=2).values  # (B,N,D)


def roi_pool_knn(
    xyz: torch.Tensor,
    feats: torch.Tensor,
    centers: torch.Tensor,
    *,
    k: int = 16,
) -> torch.Tensor:
    """Simple ROI pooling by kNN around proposal centers.

    xyz: (B,N,3)
    feats: (B,N,C)
    centers: (B,K,3)
    returns: pooled (B,K,C)
    """

    if xyz.ndim != 3 or xyz.shape[-1] != 3:
        raise ValueError(f"xyz must be (B,N,3), got {tuple(xyz.shape)}")
    if feats.ndim != 3:
        raise ValueError(f"feats must be (B,N,C), got {tuple(feats.shape)}")
    if centers.ndim != 3 or centers.shape[-1] != 3:
        raise ValueError(f"centers must be (B,K,3), got {tuple(centers.shape)}")
    if xyz.shape[:2] != feats.shape[:2]:
        raise ValueError("xyz and feats must align on (B,N)")
    if xyz.shape[0] != centers.shape[0]:
        raise ValueError("xyz and centers batch mismatch")

    from dlhub.pointcloud.ops import index_points, knn_query

    idx = knn_query(int(k), xyz, centers)  # (B,K,k)
    pooled = index_points(feats, idx).mean(dim=2)  # (B,K,C)
    return pooled


def sinusoidal_positional_encoding(x: torch.Tensor, *, num_feats: int = 64) -> torch.Tensor:
    """Sinusoidal encoding for coordinates.

    x: (..., D)
    returns: (..., 2*num_feats*D)
    """

    num_feats = int(num_feats)
    if num_feats <= 0:
        raise ValueError("num_feats must be > 0")

    x = x.to(torch.float32)
    dim = x.shape[-1]
    freqs = torch.arange(num_feats, device=x.device, dtype=x.dtype)
    freqs = 2 ** (freqs * (math.log(10000.0) / max(1, num_feats - 1)))
    freqs = freqs.view(*([1] * (x.ndim - 1)), num_feats)  # (..., F)
    x = x.unsqueeze(-1) / freqs  # (..., D, F)
    return torch.cat([x.sin(), x.cos()], dim=-1).reshape(*x.shape[:-2], dim * num_feats * 2)


class TinyTransformerEncoder(nn.Module):
    """Small transformer encoder for point tokens."""

    def __init__(
        self, d_model: int, *, nhead: int = 4, num_layers: int = 2, dropout: float = 0.0
    ) -> None:
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=int(d_model),
            nhead=int(nhead),
            dim_feedforward=int(d_model) * 4,
            dropout=float(dropout),
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=int(num_layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,N,D)
        return self.encoder(x)


def scatter_mean_2d(
    idx_hw: torch.Tensor, values: torch.Tensor, *, h: int, w: int, fill: float = 0.0
) -> torch.Tensor:
    """Scatter mean into a BEV canvas.

    idx_hw: (B,N,2) int/long with (iy, ix) indices.
    values: (B,N,C)
    returns: (B,C,H,W)
    """

    b, n, _ = idx_hw.shape
    _, _, c = values.shape
    device = values.device
    dtype = values.dtype

    idx_hw = idx_hw.to(torch.long)
    iy = idx_hw[..., 0].clamp(0, h - 1)
    ix = idx_hw[..., 1].clamp(0, w - 1)

    flat = (iy * w + ix).view(b, n)  # (B,N)
    out = torch.zeros(b, h * w, c, device=device, dtype=dtype)
    cnt = torch.zeros(b, h * w, 1, device=device, dtype=dtype)

    out.scatter_add_(1, flat.unsqueeze(-1).expand(b, n, c), values)
    cnt.scatter_add_(1, flat.unsqueeze(-1), torch.ones(b, n, 1, device=device, dtype=dtype))

    out = out / cnt.clamp_min(1.0)
    if fill != 0.0:
        out = torch.where(cnt > 0, out, torch.full_like(out, float(fill)))

    out = out.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()
    return out


@dataclass(frozen=True)
class BEVBoxSpec:
    """Simple BEV grid spec used by toy voxel/pillar detectors."""

    x_min: float = -10.0
    x_max: float = 10.0
    y_min: float = -10.0
    y_max: float = 10.0
    h: int = 32
    w: int = 32

    def quantize_xy(self, xy: torch.Tensor) -> torch.Tensor:
        # xy: (B,N,2)
        x = xy[..., 0]
        y = xy[..., 1]
        ix = ((x - self.x_min) / (self.x_max - self.x_min) * float(self.w)).floor()
        iy = ((y - self.y_min) / (self.y_max - self.y_min) * float(self.h)).floor()
        return torch.stack([iy, ix], dim=-1)


class TinyBEVBackbone(nn.Module):
    def __init__(self, in_channels: int, *, width: int = 64) -> None:
        super().__init__()
        width = int(width)
        self.net = nn.Sequential(
            nn.Conv2d(int(in_channels), width, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, width, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, width, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DenseBEVHead(nn.Module):
    """Anchor-free dense head on BEV (heatmap + box params)."""

    def __init__(self, in_channels: int, num_classes: int, *, with_yaw: bool = True) -> None:
        super().__init__()
        self.with_yaw = bool(with_yaw)
        self.heatmap = nn.Conv2d(int(in_channels), int(num_classes), 1)
        box_dim = 7 if self.with_yaw else 6
        # box: (x, y, z, dx, dy, dz, yaw?)
        self.box = nn.Conv2d(int(in_channels), box_dim, 1)

    def forward(self, feat: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"heatmap": self.heatmap(feat), "box_params": self.box(feat)}


def topk_heatmap(
    heatmap: torch.Tensor, *, k: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pick top-k positions per batch from heatmap.

    heatmap: (B,C,H,W)
    returns: scores (B,K), cls (B,K), iy (B,K), ix (B,K)
    """

    b, c, h, w = heatmap.shape
    scores = heatmap.sigmoid()
    scores = scores.view(b, c, h * w)
    topk_scores, topk_idx = scores.topk(int(k), dim=-1)  # (B,C,K)

    # Best across classes
    topk_scores2, cls = topk_scores.max(dim=1)  # (B,K)
    idx2 = topk_scores.argmax(dim=1)  # (B,K) selects which class's index to use

    flat_idx = topk_idx.gather(1, idx2.unsqueeze(1)).squeeze(1)  # (B,K)
    iy = (flat_idx // w).to(torch.long)
    ix = (flat_idx % w).to(torch.long)
    return topk_scores2, cls.to(torch.long), iy, ix


def gather_bev(feat: torch.Tensor, iy: torch.Tensor, ix: torch.Tensor) -> torch.Tensor:
    """Gather BEV features at positions.

    feat: (B,C,H,W)
    iy/ix: (B,K)
    returns: (B,K,C)
    """

    b, c, h, w = feat.shape
    k = iy.shape[1]
    iy = iy.clamp(0, h - 1)
    ix = ix.clamp(0, w - 1)
    flat = (iy * w + ix).view(b, 1, k).expand(b, c, k)  # (B,C,K)
    feat_flat = feat.view(b, c, h * w)
    out = feat_flat.gather(2, flat).permute(0, 2, 1).contiguous()  # (B,K,C)
    return out


def decode_bev_boxes(
    box_params: torch.Tensor,
    iy: torch.Tensor,
    ix: torch.Tensor,
    spec: BEVBoxSpec,
    *,
    with_yaw: bool = True,
) -> torch.Tensor:
    """Decode dense box params at top-k positions to metric boxes.

    box_params: (B,6/7,H,W)
    returns: boxes (B,K,6/7) where x/y are mapped to spec range.
    """

    b, d, h, w = box_params.shape
    iy.shape[1]
    gathered = gather_bev(box_params, iy, ix)  # (B,K,D)

    # Map grid centers to xy in metric space.
    x = (ix.float() + 0.5) / float(spec.w) * (spec.x_max - spec.x_min) + spec.x_min
    y = (iy.float() + 0.5) / float(spec.h) * (spec.y_max - spec.y_min) + spec.y_min

    # Use gathered deltas for z/dims/yaw; keep them small and positive for sizes.
    z = gathered[..., 2:3]
    dims = F.softplus(gathered[..., 3:6]) + 0.1
    if with_yaw:
        yaw = gathered[..., 6:7].tanh() * math.pi
        return torch.cat([x.unsqueeze(-1), y.unsqueeze(-1), z, dims, yaw], dim=-1)
    return torch.cat([x.unsqueeze(-1), y.unsqueeze(-1), z, dims], dim=-1)


class QueryHead(nn.Module):
    """DETR-like query head for 3D detection (toy).

    Produces fixed K boxes from global context.
    """

    def __init__(
        self, d_model: int, num_classes: int, *, num_queries: int, with_yaw: bool = True
    ) -> None:
        super().__init__()
        self.num_queries = int(num_queries)
        self.with_yaw = bool(with_yaw)
        self.query = nn.Parameter(torch.randn(self.num_queries, int(d_model)) / math.sqrt(d_model))
        self.cls = nn.Linear(int(d_model), int(num_classes))
        box_dim = 7 if self.with_yaw else 6
        self.box = nn.Linear(int(d_model), box_dim)

    def forward(self, context: torch.Tensor) -> dict[str, torch.Tensor]:
        # context: (B,D) global
        b, d = context.shape
        q = self.query.unsqueeze(0).expand(b, -1, -1)  # (B,K,D)
        q = q + context.unsqueeze(1)
        cls_logits = self.cls(q)  # (B,K,C)
        raw = self.box(q)  # (B,K,6/7)
        xyz = raw[..., :3].tanh() * 10.0
        dims = F.softplus(raw[..., 3:6]) + 0.1
        if self.with_yaw:
            yaw = raw[..., 6:7].tanh() * math.pi
            boxes = torch.cat([xyz, dims, yaw], dim=-1)
        else:
            boxes = torch.cat([xyz, dims], dim=-1)
        return {"boxes": boxes, "cls_logits": cls_logits}


class BEVAnchorFreeDetector3D(nn.Module):
    """Toy BEV detector: points -> BEV -> dense head -> top-k boxes."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        width: int,
        bev: BEVBoxSpec,
        topk: int = 64,
        with_yaw: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.bev = bev
        self.topk = int(topk)
        self.with_yaw = bool(with_yaw)

        self.point = PointNetEncoder(int(in_channels), width=int(width), dropout=float(dropout))
        self.bev_backbone = TinyBEVBackbone(int(width), width=int(width))
        self.head = DenseBEVHead(int(width), int(num_classes), with_yaw=self.with_yaw)

    def forward(self, points: torch.Tensor) -> dict[str, torch.Tensor]:
        check_points(points)
        xyz, feats = split_xyz_features(points)
        if feats is None:
            feats = xyz
        x = torch.cat([xyz, feats], dim=-1) if feats is not xyz else xyz
        p = self.point(x)  # (B,N,D)

        idx = self.bev.quantize_xy(xyz[..., :2])  # (B,N,2)
        bev = scatter_mean_2d(idx, p, h=int(self.bev.h), w=int(self.bev.w))  # (B,D,H,W)
        feat = self.bev_backbone(bev)
        dense = self.head(feat)

        scores, cls, iy, ix = topk_heatmap(dense["heatmap"], k=self.topk)
        boxes = decode_bev_boxes(dense["box_params"], iy, ix, self.bev, with_yaw=self.with_yaw)

        # One-hot class logits as a simple baseline (keeps shapes consistent).
        cls_logits = torch.zeros(
            points.shape[0],
            self.topk,
            dense["heatmap"].shape[1],
            device=points.device,
            dtype=points.dtype,
        )
        cls_logits.scatter_(-1, cls.unsqueeze(-1), scores.unsqueeze(-1))
        return {"boxes": boxes, "cls_logits": cls_logits, "scores": scores}


class BEVTwoStageDetector3D(nn.Module):
    """Toy two-stage detector: dense BEV proposals + point ROI refinement."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        width: int,
        bev: BEVBoxSpec,
        topk: int = 64,
        roi_k: int = 16,
        with_yaw: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.stage1 = BEVAnchorFreeDetector3D(
            in_channels=int(in_channels),
            num_classes=int(num_classes),
            width=int(width),
            bev=bev,
            topk=int(topk),
            with_yaw=bool(with_yaw),
            dropout=float(dropout),
        )
        self.roi_k = int(roi_k)
        d = int(width)
        self.refine = mlp(d, [d, d], d, dropout=float(dropout))
        self.cls = nn.Linear(d, int(num_classes))
        box_dim = 7 if with_yaw else 6
        self.box = nn.Linear(d, box_dim)
        self.with_yaw = bool(with_yaw)

    def forward(self, points: torch.Tensor) -> dict[str, torch.Tensor]:
        check_points(points)
        xyz, feats = split_xyz_features(points)
        if feats is None:
            feats = xyz
        x = torch.cat([xyz, feats], dim=-1) if feats is not xyz else xyz
        p = self.stage1.point(x)

        out1 = self.stage1(points)
        boxes = out1["boxes"]
        centers = boxes[..., :3]
        pooled = roi_pool_knn(xyz, p, centers, k=self.roi_k)
        r = self.refine(pooled)
        cls_logits = self.cls(r)

        raw = self.box(r)
        delta_xyz = raw[..., :3].tanh()
        delta_dims = raw[..., 3:6].tanh()
        new_xyz = boxes[..., :3] + delta_xyz
        new_dims = (boxes[..., 3:6] * (1.0 + 0.1 * delta_dims)).clamp_min(0.05)
        if self.with_yaw:
            yaw_base = boxes[..., 6:7] if boxes.shape[-1] == 7 else torch.zeros_like(raw[..., 6:7])
            new_yaw = (yaw_base + raw[..., 6:7].tanh() * 0.1).clamp(-math.pi, math.pi)
            boxes2 = torch.cat([new_xyz, new_dims, new_yaw], dim=-1)
        else:
            boxes2 = torch.cat([new_xyz, new_dims], dim=-1)

        return {"boxes": boxes2, "cls_logits": cls_logits, "scores": out1.get("scores")}


class PointQueryDetector3D(nn.Module):
    """Point tokens + optional transformer + query head (toy 3DETR style)."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        d_model: int,
        num_queries: int,
        use_transformer: bool = True,
        dropout: float = 0.0,
        with_yaw: bool = True,
    ) -> None:
        super().__init__()
        self.with_yaw = bool(with_yaw)
        self.point = PointNetEncoder(int(in_channels), width=int(d_model), dropout=float(dropout))
        self.use_transformer = bool(use_transformer)
        self._pe_num_feats = 16
        self.pe_proj = nn.Linear(3 * self._pe_num_feats * 2, int(d_model))
        self.encoder = (
            TinyTransformerEncoder(int(d_model), nhead=4, num_layers=2, dropout=float(dropout))
            if self.use_transformer
            else nn.Identity()
        )
        self.head = QueryHead(
            int(d_model), int(num_classes), num_queries=int(num_queries), with_yaw=self.with_yaw
        )

    def forward(self, points: torch.Tensor) -> dict[str, torch.Tensor]:
        check_points(points)
        x = points
        tokens = self.point(x)  # (B,N,D)
        if self.use_transformer:
            xyz, _ = split_xyz_features(points)
            pe = sinusoidal_positional_encoding(xyz, num_feats=int(self._pe_num_feats)).to(
                tokens.dtype
            )
            pe = self.pe_proj(pe)
            tokens = tokens + pe
        tokens = self.encoder(tokens)
        context = tokens.mean(dim=1)
        return self.head(context)
