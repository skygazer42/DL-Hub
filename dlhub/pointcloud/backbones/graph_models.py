from dataclasses import dataclass

import torch
from torch import nn

from ..ops import edge_features, index_points, knn_indices
from .utils import ConvBNAct2d, _c, global_max_pool


class EdgeConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            ConvBNAct2d(int(in_channels), int(out_channels), act="leaky", dropout=float(dropout)),
        )

    def forward(self, edge_feat: torch.Tensor) -> torch.Tensor:
        # edge_feat: (B, N, k, 2C)
        if edge_feat.ndim != 4:
            raise ValueError(f"edge_feat must be (B, N, k, C), got {tuple(edge_feat.shape)}")
        x = edge_feat.permute(0, 3, 1, 2).contiguous()  # (B, C, N, k)
        x = self.net(x)  # (B, out, N, k)
        x = torch.max(x, dim=-1).values  # (B, out, N)
        return x


@dataclass(frozen=True)
class DGCNNConfig:
    in_channels: int
    num_classes: int
    width_mult: float
    dropout: float
    k: int = 10
    dynamic_graph: bool = True


class DGCNNClassifier(nn.Module):
    """A small DGCNN-style classifier for toy point clouds."""

    def __init__(self, cfg: DGCNNConfig) -> None:
        super().__init__()
        self.cfg = cfg

        c_in = int(cfg.in_channels)
        h = _c(64, float(cfg.width_mult), min_ch=16, divisor=8)
        h2 = _c(128, float(cfg.width_mult), min_ch=32, divisor=8)

        self.edge1 = EdgeConv(in_channels=c_in * 2, out_channels=h, dropout=cfg.dropout)
        self.edge2 = EdgeConv(in_channels=h * 2, out_channels=h, dropout=cfg.dropout)
        self.edge3 = EdgeConv(in_channels=h * 2, out_channels=h2, dropout=cfg.dropout)

        self.fuse = nn.Sequential(
            nn.Conv1d(
                h + h + h2,
                _c(256, float(cfg.width_mult), min_ch=64, divisor=8),
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm1d(_c(256, float(cfg.width_mult), min_ch=64, divisor=8)),
            nn.LeakyReLU(0.2, inplace=True),
        )

        head_in = _c(256, float(cfg.width_mult), min_ch=64, divisor=8)
        self.head = nn.Sequential(
            nn.Linear(head_in, max(8, head_in // 2)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=float(cfg.dropout)),
            nn.Linear(max(8, head_in // 2), int(cfg.num_classes)),
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        if points.ndim != 3:
            raise ValueError(f"Expected points shape (B, N, C), got {tuple(points.shape)}")
        if int(points.shape[-1]) != int(self.cfg.in_channels):
            raise ValueError(
                f"Expected in_channels={self.cfg.in_channels}, got C={points.shape[-1]}"
            )

        x0 = points.to(torch.float32)  # (B, N, C)

        idx = knn_indices(x0, k=int(self.cfg.k))
        e = edge_features(x0, idx)
        x1 = self.edge1(e)  # (B, C1, N)

        x_for_graph = x1.transpose(1, 2).contiguous() if self.cfg.dynamic_graph else x0
        idx = knn_indices(x_for_graph, k=int(self.cfg.k))
        e = edge_features(x1.transpose(1, 2).contiguous(), idx)
        x2 = self.edge2(e)

        x_for_graph = x2.transpose(1, 2).contiguous() if self.cfg.dynamic_graph else x0
        idx = knn_indices(x_for_graph, k=int(self.cfg.k))
        e = edge_features(x2.transpose(1, 2).contiguous(), idx)
        x3 = self.edge3(e)

        x_cat = torch.cat([x1, x2, x3], dim=1)
        x_feat = self.fuse(x_cat)
        pooled = global_max_pool(x_feat)
        return self.head(pooled)


def build_dgcnn_classifier(
    *,
    in_channels: int,
    num_classes: int,
    num_points: int,
    width_mult: float = 1.0,
    dropout: float = 0.1,
    variant: str = "dgcnn",
) -> nn.Module:
    _ = int(num_points)
    name = str(variant).lower().strip()
    dynamic_graph = True
    if name in {"dgcnn_static", "edgeconv_static"}:
        dynamic_graph = False
    return DGCNNClassifier(
        DGCNNConfig(
            in_channels=int(in_channels),
            num_classes=int(num_classes),
            width_mult=float(width_mult),
            dropout=float(dropout),
            k=10,
            dynamic_graph=bool(dynamic_graph),
        )
    )


class GraphConvBlock(nn.Module):
    def __init__(self, dim: int, *, k: int, dropout: float) -> None:
        super().__init__()
        d = int(dim)
        self.k = int(k)
        self.msg = nn.Sequential(
            nn.Linear(d * 2, d),
            nn.ReLU(inplace=True),
            nn.Dropout(p=float(dropout)),
        )
        self.norm = nn.LayerNorm(d)

    def forward(self, xyz: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        # xyz: (B, N, 3), feat: (B, N, D)
        if xyz.ndim != 3 or xyz.shape[-1] != 3:
            raise ValueError(f"xyz must be (B, N, 3), got {tuple(xyz.shape)}")
        if feat.ndim != 3:
            raise ValueError(f"feat must be (B, N, D), got {tuple(feat.shape)}")
        if xyz.shape[:2] != feat.shape[:2]:
            raise ValueError("xyz and feat must align on (B, N)")

        y = self.norm(feat)
        idx = knn_indices(xyz, k=int(self.k))  # (B, N, k)
        nbr = index_points(y, idx)  # (B, N, k, D)
        agg = nbr.mean(dim=2)  # (B, N, D)
        out = self.msg(torch.cat([y, agg], dim=-1))
        return feat + out


class GraphAttentionBlock(nn.Module):
    def __init__(self, dim: int, *, k: int, dropout: float) -> None:
        super().__init__()
        d = int(dim)
        self.k = int(k)
        self.norm = nn.LayerNorm(d)
        self.q = nn.Linear(d, d, bias=False)
        self.kv = nn.Linear(d, 2 * d, bias=False)
        self.rel = nn.Sequential(
            nn.Linear(3, d),
            nn.ReLU(inplace=True),
            nn.Linear(d, d),
        )
        self.proj = nn.Sequential(
            nn.Linear(d, d, bias=False),
            nn.Dropout(p=float(dropout)),
        )

    def forward(self, xyz: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        if xyz.shape[:2] != feat.shape[:2]:
            raise ValueError("xyz and feat must align on (B, N)")

        y = self.norm(feat)
        idx = knn_indices(xyz, k=int(self.k))
        nbr_feat = index_points(y, idx)  # (B, N, k, D)
        nbr_xyz = index_points(xyz, idx)  # (B, N, k, 3)
        rel = (nbr_xyz - xyz.unsqueeze(2)).to(torch.float32)

        q = self.q(y).unsqueeze(2)  # (B, N, 1, D)
        kv = self.kv(nbr_feat)  # (B, N, k, 2D)
        k, v = kv.chunk(2, dim=-1)
        rel_e = self.rel(rel)

        attn = (q * (k + rel_e)).sum(dim=-1)  # (B, N, k)
        attn = torch.softmax(attn, dim=2)
        out = (attn.unsqueeze(-1) * (v + rel_e)).sum(dim=2)  # (B, N, D)
        return feat + self.proj(out)


class PointWebBlock(nn.Module):
    def __init__(self, dim: int, *, k: int, dropout: float) -> None:
        super().__init__()
        d = int(dim)
        self.k = int(k)
        self.norm = nn.LayerNorm(d)
        self.rel_mlp = nn.Sequential(
            nn.Linear(d * 2 + 3, d),
            nn.ReLU(inplace=True),
            nn.Dropout(p=float(dropout)),
            nn.Linear(d, 1),
        )
        self.proj = nn.Linear(d, d, bias=False)

    def forward(self, xyz: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        if xyz.shape[:2] != feat.shape[:2]:
            raise ValueError("xyz and feat must align on (B, N)")

        y = self.norm(feat)
        idx = knn_indices(xyz, k=int(self.k))
        nbr_feat = index_points(y, idx)  # (B, N, k, D)
        nbr_xyz = index_points(xyz, idx)
        rel = (nbr_xyz - xyz.unsqueeze(2)).to(torch.float32)

        c = y.unsqueeze(2).expand_as(nbr_feat)
        w = self.rel_mlp(torch.cat([c, nbr_feat, rel], dim=-1)).squeeze(-1)  # (B, N, k)
        w = torch.softmax(w, dim=2)
        delta = (w.unsqueeze(-1) * (nbr_feat - c)).sum(dim=2)  # (B, N, D)
        return feat + self.proj(delta)


@dataclass(frozen=True)
class GraphNetConfig:
    in_channels: int
    num_classes: int
    width_mult: float
    dropout: float
    embed_dim: int
    depth: int
    k: int
    block: str  # gcn|gat|pointweb


class GraphNetClassifier(nn.Module):
    def __init__(self, cfg: GraphNetConfig) -> None:
        super().__init__()
        self.cfg = cfg
        c_in = int(cfg.in_channels)
        d = _c(int(cfg.embed_dim), float(cfg.width_mult), min_ch=64, divisor=8)

        self.embed = nn.Sequential(
            nn.Linear(c_in, d),
            nn.ReLU(inplace=True),
        )

        blocks: list[nn.Module] = []
        for _ in range(int(cfg.depth)):
            if str(cfg.block) == "gcn":
                blocks.append(GraphConvBlock(d, k=int(cfg.k), dropout=float(cfg.dropout)))
            elif str(cfg.block) == "gat":
                blocks.append(GraphAttentionBlock(d, k=int(cfg.k), dropout=float(cfg.dropout)))
            elif str(cfg.block) == "pointweb":
                blocks.append(PointWebBlock(d, k=int(cfg.k), dropout=float(cfg.dropout)))
            else:
                raise ValueError("Unknown graph block type")
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Sequential(
            nn.Linear(d, d),
            nn.ReLU(inplace=True),
            nn.Dropout(p=float(cfg.dropout)),
            nn.Linear(d, int(cfg.num_classes)),
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        if points.ndim != 3 or int(points.shape[-1]) != int(self.cfg.in_channels):
            raise ValueError(
                f"Expected points shape (B, N, C={self.cfg.in_channels}), got {tuple(points.shape)}"
            )
        xyz = points[..., :3].to(torch.float32)
        feat = self.embed(points.to(torch.float32))
        for blk in self.blocks:
            feat = blk(xyz, feat)
        feat = self.norm(feat)
        pooled = feat.max(dim=1).values
        return self.head(pooled)


def build_pointgcn_classifier(
    *,
    in_channels: int,
    num_classes: int,
    num_points: int,
    width_mult: float = 1.0,
    dropout: float = 0.1,
    variant: str = "pointgcn",
) -> nn.Module:
    _ = int(num_points)
    name = str(variant).lower().strip()
    if name in {"pointgcn", "gcn"}:
        embed_dim, depth, k = 128, 4, 16
    elif name in {"pointgcn_small"}:
        embed_dim, depth, k = 160, 5, 20
    else:
        raise ValueError("Unknown PointGCN variant. Supported: pointgcn|pointgcn_small")

    return GraphNetClassifier(
        GraphNetConfig(
            in_channels=int(in_channels),
            num_classes=int(num_classes),
            width_mult=float(width_mult),
            dropout=float(dropout),
            embed_dim=int(embed_dim),
            depth=int(depth),
            k=int(k),
            block="gcn",
        )
    )


def build_pointgat_classifier(
    *,
    in_channels: int,
    num_classes: int,
    num_points: int,
    width_mult: float = 1.0,
    dropout: float = 0.1,
    variant: str = "pointgat",
) -> nn.Module:
    _ = int(num_points)
    name = str(variant).lower().strip()
    if name in {"pointgat", "gat"}:
        embed_dim, depth, k = 128, 3, 16
    elif name in {"pointgat_small"}:
        embed_dim, depth, k = 160, 4, 20
    else:
        raise ValueError("Unknown PointGAT variant. Supported: pointgat|pointgat_small")

    return GraphNetClassifier(
        GraphNetConfig(
            in_channels=int(in_channels),
            num_classes=int(num_classes),
            width_mult=float(width_mult),
            dropout=float(dropout),
            embed_dim=int(embed_dim),
            depth=int(depth),
            k=int(k),
            block="gat",
        )
    )


def build_pointweb_classifier(
    *,
    in_channels: int,
    num_classes: int,
    num_points: int,
    width_mult: float = 1.0,
    dropout: float = 0.1,
    variant: str = "pointweb",
) -> nn.Module:
    _ = int(num_points)
    name = str(variant).lower().strip()
    if name in {"pointweb", "pointweb_tiny"}:
        embed_dim, depth, k = 128, 3, 16
    elif name in {"pointweb_small"}:
        embed_dim, depth, k = 160, 4, 20
    else:
        raise ValueError("Unknown PointWeb variant. Supported: pointweb|pointweb_small")

    return GraphNetClassifier(
        GraphNetConfig(
            in_channels=int(in_channels),
            num_classes=int(num_classes),
            width_mult=float(width_mult),
            dropout=float(dropout),
            embed_dim=int(embed_dim),
            depth=int(depth),
            k=int(k),
            block="pointweb",
        )
    )


__all__ = [
    "build_dgcnn_classifier",
    "build_pointgat_classifier",
    "build_pointgcn_classifier",
    "build_pointweb_classifier",
]
