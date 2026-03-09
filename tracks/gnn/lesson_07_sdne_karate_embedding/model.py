
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    num_nodes: int
    embed_dim: int = 16
    hidden_dim: int = 64
    dropout: float = 0.1


class SDNE(nn.Module):
    """A small SDNE-style autoencoder for adjacency rows."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        n = int(cfg.num_nodes)
        h = int(cfg.hidden_dim)
        d = int(cfg.embed_dim)

        self.encoder = nn.Sequential(
            nn.Linear(n, h),
            nn.ReLU(),
            nn.Dropout(p=float(cfg.dropout)),
            nn.Linear(h, d),
        )
        self.decoder = nn.Sequential(
            nn.Linear(d, h),
            nn.ReLU(),
            nn.Dropout(p=float(cfg.dropout)),
            nn.Linear(h, n),
        )

    def forward(self, adj: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return `(recon_logits, embeddings)`."""

        if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
            raise ValueError(f"Expected adj shape (N, N), got {tuple(adj.shape)}")
        if adj.shape[0] != int(self.cfg.num_nodes):
            raise ValueError("adj N must match cfg.num_nodes")

        z = self.encoder(adj)
        recon_logits = self.decoder(z)
        return recon_logits, z


def sdne_loss(
    *,
    recon_logits: torch.Tensor,
    adj: torch.Tensor,
    embeddings: torch.Tensor,
    edge_index: torch.Tensor,
    lambda_smooth: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return `(total, recon, smooth)`."""

    if recon_logits.shape != adj.shape:
        raise ValueError("recon_logits and adj must have the same shape")

    pos = float(adj.sum().item())
    total = float(adj.numel())
    neg = max(1.0, total - pos)
    pos_weight = torch.tensor(neg / max(1.0, pos), device=adj.device, dtype=adj.dtype)

    bce = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    recon = bce(recon_logits, adj)

    src, dst = edge_index
    diff = embeddings[src] - embeddings[dst]
    smooth = (diff.pow(2).sum(dim=1)).mean()

    total_loss = recon + float(lambda_smooth) * smooth
    return total_loss, recon, smooth


__all__ = ["ModelConfig", "SDNE", "sdne_loss"]

