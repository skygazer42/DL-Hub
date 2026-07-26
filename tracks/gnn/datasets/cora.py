from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from dlhub.paths import get_repo_root

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class CoraData:
    """Cora node-classification tensors (small, classic citation network)."""

    features: torch.Tensor  # (N, F) float32
    labels: torch.Tensor  # (N,) int64

    # Sparse adjacency matrices with self-loops.
    # - `adj` uses symmetric normalization: D^{-1/2} A D^{-1/2}  (GCN-friendly)
    # - `adj_row` uses row normalization: D^{-1} A              (mean aggregator-friendly)
    adj: torch.Tensor  # (N, N) sparse float32
    adj_row: torch.Tensor  # (N, N) sparse float32

    # Standard splits used by many tutorials.
    idx_train: torch.Tensor
    idx_val: torch.Tensor
    idx_test: torch.Tensor


def _cora_dir() -> Path:
    return get_repo_root() / "tracks" / "gnn" / "assets" / "cora"


def _row_normalize(x: np.ndarray) -> np.ndarray:
    rowsum = x.sum(axis=1, keepdims=True)
    rowsum[rowsum == 0] = 1.0
    return x / rowsum


def load_cora(*, dataset_dir: str | Path | None = None) -> CoraData:
    """Load Cora from repo assets and return tensors for node classification.

    Notes:
    - Uses the commonly cited fixed split (first 140 train, 200–500 val, 500–1500 test).
    - Produces both GCN-style symmetric normalization and row-normalization.
    """

    import torch

    root = Path(dataset_dir) if dataset_dir is not None else _cora_dir()
    content_path = root / "cora.content"
    cites_path = root / "cora.cites"

    idx_features_labels = np.genfromtxt(content_path, dtype=np.dtype(str))
    paper_ids = idx_features_labels[:, 0].astype(np.int64)
    features = idx_features_labels[:, 1:-1].astype(np.float32)
    labels_str = idx_features_labels[:, -1]

    # Deterministic label mapping.
    classes = sorted(set(labels_str.tolist()))
    class_to_id = {c: i for i, c in enumerate(classes)}
    labels = np.array([class_to_id[c] for c in labels_str], dtype=np.int64)

    id_to_index = {int(pid): i for i, pid in enumerate(paper_ids.tolist())}

    edges_unordered = np.genfromtxt(cites_path, dtype=np.int64)
    src = edges_unordered[:, 0]
    dst = edges_unordered[:, 1]
    src_idx = np.array([id_to_index[int(s)] for s in src], dtype=np.int64)
    dst_idx = np.array([id_to_index[int(d)] for d in dst], dtype=np.int64)

    n = int(labels.shape[0])

    # Undirected edges + self loops. cora.cites contains mutually-citing
    # pairs; without dedup those edges get weight 2 after coalesce() and
    # the result is no longer the textbook 0/1-adjacency normalization.
    row = np.concatenate([src_idx, dst_idx, np.arange(n, dtype=np.int64)])
    col = np.concatenate([dst_idx, src_idx, np.arange(n, dtype=np.int64)])
    pairs = np.unique(np.stack([row, col], axis=1), axis=0)
    row, col = np.ascontiguousarray(pairs[:, 0]), np.ascontiguousarray(pairs[:, 1])
    val = np.ones(row.shape[0], dtype=np.float32)

    degree = np.zeros(n, dtype=np.float32)
    np.add.at(degree, row, val)

    # Symmetric normalization: D^{-1/2} A D^{-1/2}
    deg_inv_sqrt = np.power(degree, -0.5, where=degree > 0)
    deg_inv_sqrt[~np.isfinite(deg_inv_sqrt)] = 0.0
    norm_val_sym = deg_inv_sqrt[row] * val * deg_inv_sqrt[col]

    # Row normalization: D^{-1} A
    deg_inv = np.reciprocal(degree, where=degree > 0)
    deg_inv[~np.isfinite(deg_inv)] = 0.0
    norm_val_row = val * deg_inv[row]

    indices = np.vstack([row, col]).astype(np.int64)
    indices_t = torch.from_numpy(indices)
    adj = torch.sparse_coo_tensor(
        indices_t, torch.from_numpy(norm_val_sym), size=(n, n), dtype=torch.float32
    ).coalesce()
    adj_row = torch.sparse_coo_tensor(
        indices_t, torch.from_numpy(norm_val_row), size=(n, n), dtype=torch.float32
    ).coalesce()

    features = _row_normalize(features)
    features_t = torch.from_numpy(features).to(torch.float32)
    labels_t = torch.from_numpy(labels).to(torch.long)

    idx_train = torch.arange(140, dtype=torch.long)
    idx_val = torch.arange(200, 500, dtype=torch.long)
    idx_test = torch.arange(500, 1500, dtype=torch.long)

    return CoraData(
        features=features_t,
        labels=labels_t,
        adj=adj,
        adj_row=adj_row,
        idx_train=idx_train,
        idx_val=idx_val,
        idx_test=idx_test,
    )
