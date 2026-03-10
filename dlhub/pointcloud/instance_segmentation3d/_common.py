import math

import torch
from torch import nn

from dlhub.pointcloud.ops import farthest_point_sample, index_points
from dlhub.pointcloud.segmentation3d._common import (
    EdgeConv,
    FeaturePropagation,
    GridSpec2D,
    GridSpec3D,
    PointMLP,
    SetAbstraction,
    TinyTransformerEncoder,
    TinyUNet2D,
    TinyUNet3D,
    check_points,
    gather_2d,
    gather_3d,
    mlp,
    scatter_mean_2d,
    scatter_mean_3d,
    sinusoidal_positional_encoding,
    split_xyz_features,
)


def l2_normalize(x: torch.Tensor, *, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    return x / x.norm(dim=dim, keepdim=True).clamp_min(eps)


class MLPPointEncoder(nn.Module):
    def __init__(
        self, in_channels: int, width: int, *, depth: int = 3, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.mlp = PointMLP(int(in_channels), int(width), depth=int(depth), dropout=float(dropout))

    def forward(self, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        check_points(points)
        xyz, feats = split_xyz_features(points)
        x = xyz if feats is None else torch.cat([xyz, feats], dim=-1)
        feat = self.mlp(x.to(torch.float32))
        return xyz, feat


class EdgeConvEncoder(nn.Module):
    def __init__(
        self, in_channels: int, width: int, *, depth: int = 3, k: int = 16, dropout: float = 0.0
    ) -> None:
        super().__init__()
        w = int(width)
        self.embed = nn.Sequential(nn.Linear(int(in_channels), w), nn.ReLU(inplace=True))
        self.blocks = nn.ModuleList(
            [EdgeConv(w, w, k=int(k), dropout=float(dropout)) for _ in range(int(depth))]
        )

    def forward(self, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        check_points(points)
        xyz, feats = split_xyz_features(points)
        x = xyz if feats is None else torch.cat([xyz, feats], dim=-1)
        h = self.embed(x.to(torch.float32))
        for blk in self.blocks:
            h = h + blk(h)
        return xyz, h


class PointNet2Encoder(nn.Module):
    """PointNet++ style encoder returning per-point features."""

    def __init__(self, in_channels: int, width: int, *, dropout: float = 0.0) -> None:
        super().__init__()
        w = int(width)
        self.stem = PointMLP(int(in_channels), w, depth=2, dropout=float(dropout))

        self.sa1 = SetAbstraction(w, w, npoint=64, k=16, dropout=float(dropout))
        self.sa2 = SetAbstraction(w, w * 2, npoint=32, k=16, dropout=float(dropout))
        self.sa3 = SetAbstraction(w * 2, w * 4, npoint=16, k=16, dropout=float(dropout))

        self.fp2 = FeaturePropagation(w * 4 + w * 2, w * 2, dropout=float(dropout))
        self.fp1 = FeaturePropagation(w * 2 + w, w, dropout=float(dropout))
        self.fp0 = FeaturePropagation(w + w, w, dropout=float(dropout))

    def forward(self, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        check_points(points)
        xyz, feats = split_xyz_features(points)
        x = xyz if feats is None else torch.cat([xyz, feats], dim=-1)
        f0 = self.stem(x.to(torch.float32))

        xyz1, f1 = self.sa1(xyz, f0)
        xyz2, f2 = self.sa2(xyz1, f1)
        xyz3, f3 = self.sa3(xyz2, f2)

        f2u = self.fp2(xyz2, f2, xyz3, f3)
        f1u = self.fp1(xyz1, f1, xyz2, f2u)
        f0u = self.fp0(xyz, f0, xyz1, f1u)
        return xyz, f0u


class TransformerPointEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        d_model: int,
        *,
        depth: int = 3,
        dropout: float = 0.0,
        pos_feats: int = 16,
    ) -> None:
        super().__init__()
        self.embed = nn.Linear(int(in_channels), int(d_model))
        self.pos_feats = int(pos_feats)
        self.pe = nn.Linear(3 * 2 * int(pos_feats), int(d_model))
        self.enc = TinyTransformerEncoder(
            int(d_model), nhead=4, num_layers=int(depth), dropout=float(dropout)
        )

    def forward(self, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        check_points(points)
        xyz, _ = split_xyz_features(points)
        x = points.to(torch.float32)
        tok = self.embed(x)
        pe = self.pe(
            sinusoidal_positional_encoding(xyz, num_feats=int(self.pos_feats)).to(tok.dtype)
        )
        tok = tok + pe
        tok = self.enc(tok)
        return xyz, tok


class Projection2DEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        width: int,
        *,
        grid: GridSpec2D,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.grid = grid
        self.point = PointMLP(int(in_channels), int(width), depth=2, dropout=float(dropout))
        self.unet = TinyUNet2D(int(width), int(width))

    def forward(self, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        check_points(points)
        xyz, feats = split_xyz_features(points)
        x = xyz if feats is None else torch.cat([xyz, feats], dim=-1)
        p = self.point(x.to(torch.float32))
        idx = self.grid.quantize(xyz[..., :2])
        canvas = scatter_mean_2d(idx, p, h=int(self.grid.h), w=int(self.grid.w))
        feat2d = self.unet(canvas)
        gathered = gather_2d(feat2d, idx)
        return xyz, gathered


class RangeViewEncoder(nn.Module):
    """Project to a LiDAR-like range view (azimuth/elevation) then gather back to points."""

    def __init__(
        self, in_channels: int, width: int, *, h: int, w: int, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.grid = GridSpec2D(
            x_min=-math.pi,
            x_max=math.pi,
            y_min=-0.5 * math.pi,
            y_max=0.5 * math.pi,
            h=int(h),
            w=int(w),
        )
        self.point = PointMLP(int(in_channels), int(width), depth=2, dropout=float(dropout))
        self.unet = TinyUNet2D(int(width), int(width))

    def forward(self, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        check_points(points)
        xyz, feats = split_xyz_features(points)
        x = xyz if feats is None else torch.cat([xyz, feats], dim=-1)
        p = self.point(x.to(torch.float32))

        az = torch.atan2(xyz[..., 1], xyz[..., 0])
        el = torch.atan2(xyz[..., 2], (xyz[..., :2].norm(dim=-1) + 1e-6))
        idx = self.grid.quantize(torch.stack([az, el], dim=-1))
        canvas = scatter_mean_2d(idx, p, h=int(self.grid.h), w=int(self.grid.w))
        feat2d = self.unet(canvas)
        gathered = gather_2d(feat2d, idx)
        return xyz, gathered


class PolarBEVEncoder(nn.Module):
    """Project to a polar BEV (theta, r) then gather back to points."""

    def __init__(
        self,
        in_channels: int,
        width: int,
        *,
        h: int,
        w: int,
        r_max: float = 20.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.grid = GridSpec2D(
            x_min=-math.pi,
            x_max=math.pi,
            y_min=0.0,
            y_max=float(r_max),
            h=int(h),
            w=int(w),
        )
        self.point = PointMLP(int(in_channels), int(width), depth=2, dropout=float(dropout))
        self.unet = TinyUNet2D(int(width), int(width))

    def forward(self, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        check_points(points)
        xyz, feats = split_xyz_features(points)
        x = xyz if feats is None else torch.cat([xyz, feats], dim=-1)
        p = self.point(x.to(torch.float32))

        theta = torch.atan2(xyz[..., 1], xyz[..., 0])
        r = xyz[..., :2].norm(dim=-1)
        idx = self.grid.quantize(torch.stack([theta, r], dim=-1))
        canvas = scatter_mean_2d(idx, p, h=int(self.grid.h), w=int(self.grid.w))
        feat2d = self.unet(canvas)
        gathered = gather_2d(feat2d, idx)
        return xyz, gathered


class CylinderEncoder(nn.Module):
    """Project to a cylindrical view (theta, z) then gather back to points."""

    def __init__(
        self,
        in_channels: int,
        width: int,
        *,
        h: int,
        w: int,
        z_min: float = -2.0,
        z_max: float = 2.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.grid = GridSpec2D(
            x_min=-math.pi,
            x_max=math.pi,
            y_min=float(z_min),
            y_max=float(z_max),
            h=int(h),
            w=int(w),
        )
        self.point = PointMLP(int(in_channels), int(width), depth=2, dropout=float(dropout))
        self.unet = TinyUNet2D(int(width), int(width))

    def forward(self, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        check_points(points)
        xyz, feats = split_xyz_features(points)
        x = xyz if feats is None else torch.cat([xyz, feats], dim=-1)
        p = self.point(x.to(torch.float32))

        theta = torch.atan2(xyz[..., 1], xyz[..., 0])
        idx = self.grid.quantize(torch.stack([theta, xyz[..., 2]], dim=-1))
        canvas = scatter_mean_2d(idx, p, h=int(self.grid.h), w=int(self.grid.w))
        feat2d = self.unet(canvas)
        gathered = gather_2d(feat2d, idx)
        return xyz, gathered


class Voxel3DEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        width: int,
        *,
        grid: GridSpec3D,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.grid = grid
        self.point = PointMLP(int(in_channels), int(width), depth=2, dropout=float(dropout))
        self.unet = TinyUNet3D(int(width), int(width))

    def forward(self, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        check_points(points)
        xyz, feats = split_xyz_features(points)
        x = xyz if feats is None else torch.cat([xyz, feats], dim=-1)
        p = self.point(x.to(torch.float32))
        idx = self.grid.quantize(xyz)
        vol = scatter_mean_3d(idx, p, d=int(self.grid.d), h=int(self.grid.h), w=int(self.grid.w))
        feat3d = self.unet(vol)
        gathered = gather_3d(feat3d, idx)
        return xyz, gathered


class PointVoxelFusionEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        width: int,
        *,
        grid: GridSpec3D,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.grid = grid
        w = int(width)
        self.point = PointMLP(int(in_channels), w, depth=2, dropout=float(dropout))
        self.voxel = TinyUNet3D(w, w)
        self.fuse = mlp(w * 2, [w, w], w, dropout=float(dropout))

    def forward(self, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        check_points(points)
        xyz, feats = split_xyz_features(points)
        x = xyz if feats is None else torch.cat([xyz, feats], dim=-1)
        p = self.point(x.to(torch.float32))
        idx = self.grid.quantize(xyz)
        vol = scatter_mean_3d(idx, p, d=int(self.grid.d), h=int(self.grid.h), w=int(self.grid.w))
        v = self.voxel(vol)
        vg = gather_3d(v, idx)
        fused = self.fuse(torch.cat([p, vg], dim=-1))
        return xyz, fused


class QueryMaskHead(nn.Module):
    """DETR-like instance head: K learned queries -> masks via dot product."""

    def __init__(
        self, d_model: int, num_classes: int, *, num_queries: int, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.num_queries = int(num_queries)
        d = int(d_model)
        self.query = nn.Parameter(torch.randn(self.num_queries, d) / math.sqrt(d))
        self.q_proj = nn.Linear(d, d)
        self.p_proj = nn.Linear(d, d)
        self.cls = nn.Linear(d, int(num_classes))
        self.drop = nn.Dropout(float(dropout)) if dropout and dropout > 0 else nn.Identity()

    def forward(self, xyz: torch.Tensor, feat: torch.Tensor) -> dict[str, torch.Tensor]:
        # feat: (B,N,D)
        b, n, d = feat.shape
        q = self.query.unsqueeze(0).expand(b, -1, -1)  # (B,K,D)
        q = self.drop(self.q_proj(q))
        p = self.drop(self.p_proj(feat))
        qn = l2_normalize(q, dim=-1)
        pn = l2_normalize(p, dim=-1)
        mask_logits = torch.einsum("bkd,bnd->bkn", qn, pn) * math.sqrt(d)
        cls_logits = self.cls(q)
        return {"mask_logits": mask_logits, "cls_logits": cls_logits}


class PrototypeMaskHead(nn.Module):
    """Prototype head: K prototypes in feature space -> masks via similarity."""

    def __init__(
        self, d_model: int, num_classes: int, *, num_prototypes: int, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.num_prototypes = int(num_prototypes)
        d = int(d_model)
        self.proto = nn.Parameter(torch.randn(self.num_prototypes, d) / math.sqrt(d))
        self.p_proj = nn.Linear(d, d)
        self.cls = nn.Linear(d, int(num_classes))
        self.drop = nn.Dropout(float(dropout)) if dropout and dropout > 0 else nn.Identity()

    def forward(self, xyz: torch.Tensor, feat: torch.Tensor) -> dict[str, torch.Tensor]:
        b, n, d = feat.shape
        p = self.drop(self.p_proj(feat))
        pn = l2_normalize(p, dim=-1)
        proto = l2_normalize(self.proto, dim=-1).unsqueeze(0).expand(b, -1, -1)  # (B,K,D)
        mask_logits = torch.einsum("bkd,bnd->bkn", proto, pn) * math.sqrt(d)
        cls_logits = self.cls(proto)
        return {"mask_logits": mask_logits, "cls_logits": cls_logits}


class YOLACTHead(nn.Module):
    """YOLACT-style: global prototypes (P) + per-instance coeffs -> masks."""

    def __init__(
        self,
        d_model: int,
        num_classes: int,
        *,
        num_instances: int,
        num_prototypes: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_instances = int(num_instances)
        self.num_prototypes = int(num_prototypes)
        d = int(d_model)
        self.proto = nn.Linear(d, self.num_prototypes)
        self.inst = nn.Parameter(torch.randn(self.num_instances, d) / math.sqrt(d))
        self.coeff = nn.Linear(d, self.num_prototypes)
        self.cls = nn.Linear(d, int(num_classes))
        self.drop = nn.Dropout(float(dropout)) if dropout and dropout > 0 else nn.Identity()

    def forward(self, xyz: torch.Tensor, feat: torch.Tensor) -> dict[str, torch.Tensor]:
        # point prototypes: (B,N,P)
        proto = self.proto(self.drop(feat))
        inst = self.inst.unsqueeze(0).expand(feat.shape[0], -1, -1)
        coeff = self.coeff(self.drop(inst))  # (B,K,P)
        mask_logits = torch.einsum("bkp,bnp->bkn", coeff, proto)
        cls_logits = self.cls(inst)
        return {"mask_logits": mask_logits, "cls_logits": cls_logits}


class CenterProposalHead(nn.Module):
    """Proposal-style instance head: pick K centers (FPS) -> masks via distance + feature sim."""

    def __init__(
        self, d_model: int, num_classes: int, *, num_instances: int, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.num_instances = int(num_instances)
        d = int(d_model)
        self.center_fc = nn.Sequential(nn.Linear(d, d), nn.ReLU(inplace=True), nn.Linear(d, 3))
        self.cls = nn.Linear(d, int(num_classes))
        self.drop = nn.Dropout(float(dropout)) if dropout and dropout > 0 else nn.Identity()

    def forward(self, xyz: torch.Tensor, feat: torch.Tensor) -> dict[str, torch.Tensor]:
        b, n, d = feat.shape
        k = min(self.num_instances, n)
        idx = farthest_point_sample(xyz, k)  # (B,K)
        centers = index_points(xyz, idx)  # (B,K,3)
        inst_feat = index_points(feat, idx)  # (B,K,D)
        inst_feat = self.drop(inst_feat)

        dist2 = torch.cdist(centers.to(torch.float32), xyz.to(torch.float32)) ** 2  # (B,K,N)
        sim = torch.einsum(
            "bkd,bnd->bkn", l2_normalize(inst_feat, dim=-1), l2_normalize(feat, dim=-1)
        ) * math.sqrt(d)
        mask_logits = sim - 0.1 * dist2.to(sim.dtype)
        cls_logits = self.cls(inst_feat)
        return {"mask_logits": mask_logits, "cls_logits": cls_logits, "centers": centers}


class SimilarityPivotHead(nn.Module):
    """SGPN-ish: embedding similarity to K pivot points defines masks."""

    def __init__(
        self, d_model: int, num_classes: int, *, num_instances: int, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.num_instances = int(num_instances)
        d = int(d_model)
        self.embed = nn.Sequential(nn.Linear(d, d), nn.ReLU(inplace=True), nn.Linear(d, d))
        self.cls = nn.Linear(d, int(num_classes))
        self.drop = nn.Dropout(float(dropout)) if dropout and dropout > 0 else nn.Identity()

    def forward(self, xyz: torch.Tensor, feat: torch.Tensor) -> dict[str, torch.Tensor]:
        b, n, d = feat.shape
        k = min(self.num_instances, n)
        emb = self.embed(self.drop(feat))
        idx = farthest_point_sample(xyz, k)
        piv = index_points(emb, idx)  # (B,K,D)
        sim = torch.einsum(
            "bkd,bnd->bkn", l2_normalize(piv, dim=-1), l2_normalize(emb, dim=-1)
        ) * math.sqrt(d)
        cls_logits = self.cls(piv)
        return {"mask_logits": sim, "cls_logits": cls_logits}
