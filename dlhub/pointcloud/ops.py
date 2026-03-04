from __future__ import annotations

import torch


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Batch index points/features.

    Args:
        points: (B, N, C)
        idx: (B, S) or (B, S, K)

    Returns:
        (B, S, C) or (B, S, K, C)
    """

    if points.ndim != 3:
        raise ValueError(f"points must be (B, N, C), got {tuple(points.shape)}")
    b, _, _ = points.shape

    batch_indices = torch.arange(b, device=points.device).view(b, 1, 1)
    if idx.ndim == 2:
        batch_indices = batch_indices[:, :, 0]
        return points[batch_indices, idx, :]
    if idx.ndim == 3:
        batch_indices = batch_indices.expand(-1, idx.shape[1], idx.shape[2])
        return points[batch_indices, idx, :]
    raise ValueError(f"idx must be (B, S) or (B, S, K), got {tuple(idx.shape)}")


def knn_indices(x: torch.Tensor, k: int, *, exclude_self: bool = True) -> torch.Tensor:
    """kNN indices for a batch of point features.

    Args:
        x: (B, N, C) points/features
        k: neighbors (must satisfy 1 <= k < N when exclude_self=True)
        exclude_self: when True, removes self from nearest neighbors

    Returns:
        idx: (B, N, k)
    """

    if x.ndim != 3:
        raise ValueError(f"x must be (B, N, C), got {tuple(x.shape)}")
    b, n, _ = x.shape
    k = int(k)
    if k <= 0:
        raise ValueError("k must be > 0")
    if exclude_self and k >= n:
        raise ValueError(f"k must be < N when exclude_self=True, got k={k}, N={n}")
    if not exclude_self and k > n:
        raise ValueError(f"k must be <= N when exclude_self=False, got k={k}, N={n}")

    x = x.to(torch.float32)
    dist = torch.cdist(x, x)  # (B, N, N)
    if exclude_self:
        dist = dist + torch.eye(n, device=x.device, dtype=dist.dtype).unsqueeze(0) * 1e9
    idx = dist.topk(k=k, dim=-1, largest=False).indices
    if idx.shape != (b, n, k):
        raise RuntimeError("knn_indices produced an unexpected shape")
    return idx


def knn_query(k: int, xyz: torch.Tensor, query_xyz: torch.Tensor) -> torch.Tensor:
    """kNN search for query points.

    Args:
        xyz: (B, N, 3)
        query_xyz: (B, S, 3)

    Returns:
        idx: (B, S, k) indices into N points.
    """

    if xyz.ndim != 3 or xyz.shape[-1] != 3:
        raise ValueError(f"xyz must be (B, N, 3), got {tuple(xyz.shape)}")
    if query_xyz.ndim != 3 or query_xyz.shape[-1] != 3:
        raise ValueError(f"query_xyz must be (B, S, 3), got {tuple(query_xyz.shape)}")
    b, n, _ = xyz.shape
    b2, s, _ = query_xyz.shape
    if b2 != b:
        raise ValueError("xyz and query_xyz batch size mismatch")

    k = int(k)
    if k <= 0 or k > n:
        raise ValueError(f"k must be in [1, N], got k={k} with N={n}")

    dist = torch.cdist(query_xyz.to(torch.float32), xyz.to(torch.float32))  # (B, S, N)
    idx = dist.topk(k=k, dim=-1, largest=False).indices
    if idx.shape != (b, s, k):
        raise RuntimeError("knn_query produced an unexpected shape")
    return idx


def farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """Farthest point sampling indices (FPS).

    xyz: (B, N, 3)
    returns: (B, npoint)
    """

    if xyz.ndim != 3 or xyz.shape[-1] != 3:
        raise ValueError(f"xyz must be (B, N, 3), got {tuple(xyz.shape)}")
    b, n, _ = xyz.shape
    npoint = int(npoint)
    if npoint <= 0 or npoint > n:
        raise ValueError(f"npoint must be in [1, N], got {npoint} with N={n}")

    xyz = xyz.to(torch.float32)
    centroids = torch.zeros((b, npoint), dtype=torch.long, device=xyz.device)
    distance = torch.full((b, n), 1e10, device=xyz.device, dtype=xyz.dtype)
    # Deterministic init (important when two networks must sample aligned patches,
    # e.g. student/teacher SSL with masked patch distillation).
    centroid0 = xyz.mean(dim=1, keepdim=True)  # (B, 1, 3)
    dist0 = ((xyz - centroid0) ** 2).sum(dim=-1)  # (B, N)
    farthest = dist0.max(dim=1).indices  # (B,)
    batch_indices = torch.arange(b, device=xyz.device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(b, 1, 3)
        dist = ((xyz - centroid) ** 2).sum(dim=-1)
        mask = dist < distance
        distance = torch.where(mask, dist, distance)
        farthest = torch.max(distance, dim=1).indices

    return centroids


def edge_features(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Build EdgeConv-style features.

    Args:
        x: (B, N, C)
        idx: (B, N, k)

    Returns:
        edge: (B, N, k, 2C) where edge = concat(x_i, x_j - x_i)
    """

    if x.ndim != 3:
        raise ValueError(f"x must be (B, N, C), got {tuple(x.shape)}")
    if idx.ndim != 3:
        raise ValueError(f"idx must be (B, N, k), got {tuple(idx.shape)}")
    b, n, c = x.shape
    b2, n2, k = idx.shape
    if b2 != b or n2 != n:
        raise ValueError("idx must align with x on (B, N)")

    neighbors = index_points(x, idx)  # (B, N, k, C)
    central = x.unsqueeze(2).expand(-1, -1, k, -1)
    return torch.cat([central, neighbors - central], dim=-1)


def chamfer_distance(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute Chamfer Distance between two point clouds (batched).

    Args:
        pred: (B, N, 3)
        target: (B, M, 3)

    Returns:
        scalar Tensor (mean over batch)
    """

    if pred.ndim != 3 or pred.shape[-1] != 3:
        raise ValueError(f"pred must be (B, N, 3), got {tuple(pred.shape)}")
    if target.ndim != 3 or target.shape[-1] != 3:
        raise ValueError(f"target must be (B, M, 3), got {tuple(target.shape)}")
    if pred.shape[0] != target.shape[0]:
        raise ValueError("pred and target batch size mismatch")

    pred = pred.to(torch.float32)
    target = target.to(torch.float32)
    dist = torch.cdist(pred, target)  # (B, N, M)
    dist2 = dist * dist
    min_pred = dist2.min(dim=2).values  # (B, N)
    min_target = dist2.min(dim=1).values  # (B, M)
    return (min_pred.mean(dim=1) + min_target.mean(dim=1)).mean()


__all__ = [
    "chamfer_distance",
    "edge_features",
    "farthest_point_sample",
    "index_points",
    "knn_indices",
    "knn_query",
]
