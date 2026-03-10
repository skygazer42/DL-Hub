import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from dlhub.pointcloud.ops import farthest_point_sample, index_points
from dlhub.pointcloud.ops import knn_indices as knn_indices_ops
from dlhub.pointcloud.ops import knn_query


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


def normalize_xyz(xyz: torch.Tensor, *, eps: float = 1e-6) -> torch.Tensor:
    xyz = xyz - xyz.mean(dim=1, keepdim=True)
    xyz = xyz / xyz.std(dim=1, keepdim=True).clamp_min(eps)
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


class PointMLP(nn.Module):
    """Per-point MLP stack (B,N,C)->(B,N,D)."""

    def __init__(
        self, in_channels: int, width: int, *, depth: int = 3, dropout: float = 0.0
    ) -> None:
        super().__init__()
        c = int(in_channels)
        w = int(width)
        d = int(depth)
        layers: list[nn.Module] = []
        for i in range(d):
            layers.append(nn.Linear(c if i == 0 else w, w))
            layers.append(nn.ReLU(inplace=True))
            if dropout and dropout > 0:
                layers.append(nn.Dropout(float(dropout)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EdgeConv(nn.Module):
    """DGCNN-style EdgeConv (toy)."""

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
        if x.ndim != 3:
            raise ValueError(f"Expected (B,N,C), got {tuple(x.shape)}")
        idx = knn_indices_ops(x, int(self.k), exclude_self=True)  # (B,N,k)
        neighbors = index_points(x, idx)  # (B,N,k,C)
        x_i = x.unsqueeze(2).expand_as(neighbors)
        edge = torch.cat([x_i, neighbors - x_i], dim=-1)
        edge = self.mlp(edge)
        return edge.max(dim=2).values


class TinyTransformerEncoder(nn.Module):
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
        self.enc = nn.TransformerEncoder(layer, num_layers=int(num_layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.enc(x)


def sinusoidal_positional_encoding(xyz: torch.Tensor, *, num_feats: int = 16) -> torch.Tensor:
    """Sinusoidal encoding for xyz coordinates.

    xyz: (B,N,3)
    returns: (B,N,3*2*num_feats)
    """

    num_feats = int(num_feats)
    xyz = xyz.to(torch.float32)
    freqs = torch.arange(num_feats, device=xyz.device, dtype=xyz.dtype)
    freqs = 2 ** (freqs * (math.log(10000.0) / max(1, num_feats - 1)))
    freqs = freqs.view(1, 1, num_feats)  # (1,1,F)
    out: list[torch.Tensor] = []
    for d in range(3):
        v = xyz[..., d : d + 1] / freqs  # (B,N,F)
        out.append(v.sin())
        out.append(v.cos())
    return torch.cat(out, dim=-1)


class SetAbstraction(nn.Module):
    """PointNet++ style Set Abstraction (FPS + kNN + shared MLP + maxpool)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        npoint: int,
        k: int = 16,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.npoint = int(npoint)
        self.k = int(k)
        self.out_channels = int(out_channels)
        self.local = mlp(
            int(in_channels) + 3,
            [int(out_channels), int(out_channels)],
            int(out_channels),
            dropout=float(dropout),
        )

    def forward(self, xyz: torch.Tensor, feats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # xyz: (B,N,3), feats: (B,N,C)
        if xyz.ndim != 3 or xyz.shape[-1] != 3:
            raise ValueError(f"xyz must be (B,N,3), got {tuple(xyz.shape)}")
        if feats.ndim != 3 or feats.shape[:2] != xyz.shape[:2]:
            raise ValueError("feats must be (B,N,C) aligned with xyz")

        npoint = min(int(self.npoint), int(xyz.shape[1]))
        k = min(int(self.k), int(xyz.shape[1]))
        idx = farthest_point_sample(xyz, npoint)  # (B,S)
        new_xyz = index_points(xyz, idx)  # (B,S,3)
        neigh = knn_query(k, xyz, new_xyz)  # (B,S,k) indices into N
        neigh_xyz = index_points(xyz, neigh)  # (B,S,k,3)
        neigh_feats = index_points(feats, neigh)  # (B,S,k,C)

        rel = neigh_xyz - new_xyz.unsqueeze(2)
        x = torch.cat([rel, neigh_feats], dim=-1)  # (B,S,k,3+C)
        y = self.local(x)  # (B,S,k,D)
        y = y.max(dim=2).values  # (B,S,D)
        return new_xyz, y


class FeaturePropagation(nn.Module):
    """PointNet++ feature propagation with 3-NN interpolation."""

    def __init__(self, in_channels: int, out_channels: int, *, dropout: float = 0.0) -> None:
        super().__init__()
        self.fuse = mlp(
            int(in_channels),
            [int(out_channels), int(out_channels)],
            int(out_channels),
            dropout=float(dropout),
        )

    def forward(
        self,
        xyz1: torch.Tensor,
        feats1: torch.Tensor | None,
        xyz2: torch.Tensor,
        feats2: torch.Tensor,
        *,
        k: int = 3,
    ) -> torch.Tensor:
        # propagate feats2 (S) to xyz1 (N)
        if xyz1.ndim != 3 or xyz1.shape[-1] != 3:
            raise ValueError("xyz1 must be (B,N,3)")
        if xyz2.ndim != 3 or xyz2.shape[-1] != 3:
            raise ValueError("xyz2 must be (B,S,3)")
        if feats2.ndim != 3 or feats2.shape[:2] != xyz2.shape[:2]:
            raise ValueError("feats2 must align with xyz2")
        if feats1 is not None and (feats1.ndim != 3 or feats1.shape[:2] != xyz1.shape[:2]):
            raise ValueError("feats1 must align with xyz1")

        k = min(int(k), int(xyz2.shape[1]))
        idx = knn_query(k, xyz2, xyz1)  # (B,N,k) indices into S
        neigh_xyz = index_points(xyz2, idx)  # (B,N,k,3)
        neigh_feats = index_points(feats2, idx)  # (B,N,k,C2)
        dist = ((xyz1.unsqueeze(2) - neigh_xyz) ** 2).sum(dim=-1).clamp_min(1e-8).sqrt()  # (B,N,k)
        w = (1.0 / dist).softmax(dim=-1).unsqueeze(-1)  # (B,N,k,1)
        interp = (neigh_feats * w).sum(dim=2)  # (B,N,C2)

        x = interp if feats1 is None else torch.cat([interp, feats1], dim=-1)
        return self.fuse(x)


@dataclass(frozen=True)
class GridSpec2D:
    x_min: float = -10.0
    x_max: float = 10.0
    y_min: float = -10.0
    y_max: float = 10.0
    h: int = 32
    w: int = 32

    def quantize(self, xy: torch.Tensor) -> torch.Tensor:
        # xy: (B,N,2)
        x = xy[..., 0]
        y = xy[..., 1]
        ix = ((x - self.x_min) / (self.x_max - self.x_min) * float(self.w)).floor()
        iy = ((y - self.y_min) / (self.y_max - self.y_min) * float(self.h)).floor()
        return torch.stack([iy, ix], dim=-1)


def scatter_mean_2d(idx_hw: torch.Tensor, values: torch.Tensor, *, h: int, w: int) -> torch.Tensor:
    """Scatter mean into a (B,C,H,W) canvas."""

    b, n, _ = idx_hw.shape
    _, _, c = values.shape
    device = values.device
    dtype = values.dtype

    idx_hw = idx_hw.to(torch.long)
    iy = idx_hw[..., 0].clamp(0, h - 1)
    ix = idx_hw[..., 1].clamp(0, w - 1)
    flat = (iy * w + ix).view(b, n)

    out = torch.zeros(b, h * w, c, device=device, dtype=dtype)
    cnt = torch.zeros(b, h * w, 1, device=device, dtype=dtype)
    out.scatter_add_(1, flat.unsqueeze(-1).expand(b, n, c), values)
    cnt.scatter_add_(1, flat.unsqueeze(-1), torch.ones(b, n, 1, device=device, dtype=dtype))
    out = out / cnt.clamp_min(1.0)
    return out.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()


def gather_2d(feat: torch.Tensor, idx_hw: torch.Tensor) -> torch.Tensor:
    """Gather BEV features at point indices.

    feat: (B,C,H,W)
    idx_hw: (B,N,2)
    returns: (B,N,C)
    """

    b, c, h, w = feat.shape
    idx_hw = idx_hw.to(torch.long)
    iy = idx_hw[..., 0].clamp(0, h - 1)
    ix = idx_hw[..., 1].clamp(0, w - 1)
    flat = (iy * w + ix).view(b, 1, -1).expand(b, c, -1)  # (B,C,N)
    feat_flat = feat.view(b, c, h * w)
    out = feat_flat.gather(2, flat).permute(0, 2, 1).contiguous()
    return out


class TinyUNet2D(nn.Module):
    def __init__(self, in_channels: int, width: int) -> None:
        super().__init__()
        w = int(width)
        self.enc1 = nn.Sequential(
            nn.Conv2d(int(in_channels), w, 3, padding=1), nn.ReLU(inplace=True)
        )
        self.enc2 = nn.Sequential(nn.Conv2d(w, w, 3, stride=2, padding=1), nn.ReLU(inplace=True))
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(w, w, 4, stride=2, padding=1), nn.ReLU(inplace=True)
        )
        self.fuse = nn.Conv2d(w * 2, w, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.dec(x2)
        if x3.shape[-2:] != x1.shape[-2:]:
            x3 = F.interpolate(x3, size=x1.shape[-2:], mode="bilinear", align_corners=False)
        return F.relu(self.fuse(torch.cat([x1, x3], dim=1)), inplace=True)


@dataclass(frozen=True)
class GridSpec3D:
    x_min: float = -10.0
    x_max: float = 10.0
    y_min: float = -10.0
    y_max: float = 10.0
    z_min: float = -2.0
    z_max: float = 2.0
    d: int = 8
    h: int = 32
    w: int = 32

    def quantize(self, xyz: torch.Tensor) -> torch.Tensor:
        # xyz: (B,N,3) -> (B,N,3) as (iz,iy,ix)
        x = xyz[..., 0]
        y = xyz[..., 1]
        z = xyz[..., 2]
        ix = ((x - self.x_min) / (self.x_max - self.x_min) * float(self.w)).floor()
        iy = ((y - self.y_min) / (self.y_max - self.y_min) * float(self.h)).floor()
        iz = ((z - self.z_min) / (self.z_max - self.z_min) * float(self.d)).floor()
        return torch.stack([iz, iy, ix], dim=-1)


def scatter_mean_3d(
    idx_dhw: torch.Tensor, values: torch.Tensor, *, d: int, h: int, w: int
) -> torch.Tensor:
    """Scatter mean into a (B,C,D,H,W) volume."""

    b, n, _ = idx_dhw.shape
    _, _, c = values.shape
    device = values.device
    dtype = values.dtype

    idx_dhw = idx_dhw.to(torch.long)
    iz = idx_dhw[..., 0].clamp(0, d - 1)
    iy = idx_dhw[..., 1].clamp(0, h - 1)
    ix = idx_dhw[..., 2].clamp(0, w - 1)
    flat = (iz * (h * w) + iy * w + ix).view(b, n)  # (B,N)

    out = torch.zeros(b, d * h * w, c, device=device, dtype=dtype)
    cnt = torch.zeros(b, d * h * w, 1, device=device, dtype=dtype)
    out.scatter_add_(1, flat.unsqueeze(-1).expand(b, n, c), values)
    cnt.scatter_add_(1, flat.unsqueeze(-1), torch.ones(b, n, 1, device=device, dtype=dtype))
    out = out / cnt.clamp_min(1.0)
    return out.view(b, d, h, w, c).permute(0, 4, 1, 2, 3).contiguous()


def gather_3d(feat: torch.Tensor, idx_dhw: torch.Tensor) -> torch.Tensor:
    """Gather voxel features at point indices.

    feat: (B,C,D,H,W)
    idx_dhw: (B,N,3) as (iz,iy,ix)
    returns: (B,N,C)
    """

    b, c, d, h, w = feat.shape
    idx_dhw = idx_dhw.to(torch.long)
    iz = idx_dhw[..., 0].clamp(0, d - 1)
    iy = idx_dhw[..., 1].clamp(0, h - 1)
    ix = idx_dhw[..., 2].clamp(0, w - 1)
    flat = (iz * (h * w) + iy * w + ix).view(b, 1, -1).expand(b, c, -1)  # (B,C,N)
    feat_flat = feat.view(b, c, d * h * w)
    out = feat_flat.gather(2, flat).permute(0, 2, 1).contiguous()
    return out


class TinyUNet3D(nn.Module):
    def __init__(self, in_channels: int, width: int) -> None:
        super().__init__()
        w = int(width)
        self.enc1 = nn.Sequential(
            nn.Conv3d(int(in_channels), w, 3, padding=1), nn.ReLU(inplace=True)
        )
        self.enc2 = nn.Sequential(nn.Conv3d(w, w, 3, stride=2, padding=1), nn.ReLU(inplace=True))
        self.dec = nn.Sequential(
            nn.ConvTranspose3d(w, w, 4, stride=2, padding=1), nn.ReLU(inplace=True)
        )
        self.fuse = nn.Conv3d(w * 2, w, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.dec(x2)
        if x3.shape[-3:] != x1.shape[-3:]:
            x3 = F.interpolate(x3, size=x1.shape[-3:], mode="trilinear", align_corners=False)
        return F.relu(self.fuse(torch.cat([x1, x3], dim=1)), inplace=True)


class PointNetSegBase(nn.Module):
    def __init__(
        self, *, in_channels: int, num_classes: int, width: int, depth: int, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.enc = PointMLP(int(in_channels), int(width), depth=int(depth), dropout=float(dropout))
        self.fuse = mlp(
            int(width) * 2, [int(width), int(width)], int(width), dropout=float(dropout)
        )
        self.cls = nn.Linear(int(width), int(num_classes))

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        check_points(points)
        x = points.to(torch.float32)
        p = self.enc(x)  # (B,N,D)
        g = p.max(dim=1, keepdim=True).values.expand_as(p)
        y = self.fuse(torch.cat([p, g], dim=-1))
        return self.cls(y)


class EdgeConvSegBase(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        width: int,
        k: int,
        depth: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        d = int(depth)
        w = int(width)
        layers: list[nn.Module] = []
        c_in = int(in_channels)
        for i in range(d):
            layers.append(EdgeConv(c_in if i == 0 else w, w, k=int(k), dropout=float(dropout)))
        self.layers = nn.ModuleList(layers)
        self.fuse = nn.Sequential(nn.Linear(w * d, w), nn.ReLU(inplace=True))
        self.cls = nn.Linear(w, int(num_classes))

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        check_points(points)
        xyz, feats = split_xyz_features(points)
        x = xyz if feats is None else torch.cat([xyz, feats], dim=-1)
        feats_list: list[torch.Tensor] = []
        h = x
        for layer in self.layers:
            h = layer(h)
            feats_list.append(h)
        y = self.fuse(torch.cat(feats_list, dim=-1))
        return self.cls(y)


class PointNet2SegBase(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        width: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        w = int(width)
        self.stem = PointMLP(int(in_channels), w, depth=2, dropout=float(dropout))

        self.sa1 = SetAbstraction(w, w, npoint=64, k=16, dropout=float(dropout))
        self.sa2 = SetAbstraction(w, w * 2, npoint=32, k=16, dropout=float(dropout))
        self.sa3 = SetAbstraction(w * 2, w * 4, npoint=16, k=16, dropout=float(dropout))

        self.fp2 = FeaturePropagation(w * 4 + w * 2, w * 2, dropout=float(dropout))
        self.fp1 = FeaturePropagation(w * 2 + w, w, dropout=float(dropout))
        self.fp0 = FeaturePropagation(w + w, w, dropout=float(dropout))

        self.cls = nn.Sequential(
            nn.Linear(w, w), nn.ReLU(inplace=True), nn.Linear(w, int(num_classes))
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        check_points(points)
        xyz, feats = split_xyz_features(points)
        x = xyz if feats is None else torch.cat([xyz, feats], dim=-1)
        f0 = self.stem(x)

        xyz1, f1 = self.sa1(xyz, f0)
        xyz2, f2 = self.sa2(xyz1, f1)
        xyz3, f3 = self.sa3(xyz2, f2)

        f2u = self.fp2(xyz2, f2, xyz3, f3, k=3)
        f1u = self.fp1(xyz1, f1, xyz2, f2u, k=3)
        f0u = self.fp0(xyz, f0, xyz1, f1u, k=3)
        return self.cls(f0u)


class TransformerSegBase(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        d_model: int,
        depth: int,
        dropout: float = 0.0,
        pos_feats: int = 16,
    ) -> None:
        super().__init__()
        self.embed = nn.Linear(int(in_channels), int(d_model))
        self.pe = nn.Linear(3 * 2 * int(pos_feats), int(d_model))
        self.enc = TinyTransformerEncoder(
            int(d_model), nhead=4, num_layers=int(depth), dropout=float(dropout)
        )
        self.cls = nn.Sequential(
            nn.Linear(int(d_model), int(d_model)),
            nn.ReLU(inplace=True),
            nn.Linear(int(d_model), int(num_classes)),
        )
        self.pos_feats = int(pos_feats)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        check_points(points)
        xyz, _ = split_xyz_features(points)
        x = points.to(torch.float32)
        tok = self.embed(x)
        pe = self.pe(
            sinusoidal_positional_encoding(xyz, num_feats=int(self.pos_feats)).to(tok.dtype)
        )
        tok = tok + pe
        tok = self.enc(tok)
        return self.cls(tok)


class Projection2DSegBase(nn.Module):
    """Project points to a 2D grid, run a tiny UNet2D, then gather back to points."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        width: int,
        grid: GridSpec2D,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.grid = grid
        self.point = PointMLP(int(in_channels), int(width), depth=2, dropout=float(dropout))
        self.unet = TinyUNet2D(int(width), int(width))
        self.cls = nn.Linear(int(width), int(num_classes))

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        check_points(points)
        xyz, feats = split_xyz_features(points)
        x = xyz if feats is None else torch.cat([xyz, feats], dim=-1)
        p = self.point(x)
        idx = self.grid.quantize(xyz[..., :2])
        bev = scatter_mean_2d(idx, p, h=int(self.grid.h), w=int(self.grid.w))
        feat = self.unet(bev)
        gathered = gather_2d(feat, idx)
        return self.cls(gathered)


class Voxel3DSegBase(nn.Module):
    """Voxelize into a 3D grid, run a tiny UNet3D, gather back to points."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        width: int,
        grid: GridSpec3D,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.grid = grid
        self.point = PointMLP(int(in_channels), int(width), depth=2, dropout=float(dropout))
        self.unet = TinyUNet3D(int(width), int(width))
        self.cls = nn.Linear(int(width), int(num_classes))

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        check_points(points)
        xyz, feats = split_xyz_features(points)
        x = xyz if feats is None else torch.cat([xyz, feats], dim=-1)
        p = self.point(x)
        idx = self.grid.quantize(xyz)
        vox = scatter_mean_3d(idx, p, d=int(self.grid.d), h=int(self.grid.h), w=int(self.grid.w))
        feat = self.unet(vox)
        gathered = gather_3d(feat, idx)
        return self.cls(gathered)


class PointVoxelFusionSegBase(nn.Module):
    """Point-voxel fusion backbone (toy SPVCNN/PVCNN style)."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        width: int,
        grid: GridSpec3D,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.grid = grid
        w = int(width)
        self.point = PointMLP(int(in_channels), w, depth=2, dropout=float(dropout))
        self.voxel = TinyUNet3D(w, w)
        self.fuse = mlp(w * 2, [w, w], w, dropout=float(dropout))
        self.cls = nn.Linear(w, int(num_classes))

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        check_points(points)
        xyz, feats = split_xyz_features(points)
        x = xyz if feats is None else torch.cat([xyz, feats], dim=-1)
        p = self.point(x)  # (B,N,W)
        idx = self.grid.quantize(xyz)
        vox = scatter_mean_3d(idx, p, d=int(self.grid.d), h=int(self.grid.h), w=int(self.grid.w))
        v = self.voxel(vox)
        vg = gather_3d(v, idx)  # (B,N,W)
        y = self.fuse(torch.cat([p, vg], dim=-1))
        return self.cls(y)
