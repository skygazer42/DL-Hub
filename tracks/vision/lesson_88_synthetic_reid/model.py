from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    in_channels: int = 3
    num_classes: int = 8
    arch: str = "osnet:osnet_tiny"
    variant: str = ""
    width_mult: float = 1.0
    dropout: float = 0.0


def list_supported_arches() -> list[str]:
    from dlhub.vision.reid.agw import _VARIANTS as agw_variants
    from dlhub.vision.reid.fastreid import _VARIANTS as fastreid_variants
    from dlhub.vision.reid.osnet import _VARIANTS as osnet_variants
    from dlhub.vision.reid.transreid import _VARIANTS as transreid_variants

    return (
        [f"agw:{name}" for name in sorted(agw_variants)]
        + [f"fastreid:{name}" for name in sorted(fastreid_variants)]
        + [f"osnet:{name}" for name in sorted(osnet_variants)]
        + [f"transreid:{name}" for name in sorted(transreid_variants)]
        + ["agw", "fastreid", "osnet", "transreid"]
    )


def build_model(cfg: ModelConfig) -> nn.Module:
    arch_raw = str(cfg.arch).strip()
    arch = arch_raw.lower()
    variant = str(cfg.variant).strip()
    if ":" in arch_raw:
        prefix, name = arch_raw.split(":", 1)
        arch = prefix.strip().lower()
        variant = name.strip()

    kwargs = {
        "in_channels": int(cfg.in_channels),
        "num_classes": int(cfg.num_classes),
        "variant": str(variant) if variant else "",
        "width_mult": float(cfg.width_mult),
        "dropout": float(cfg.dropout),
    }

    if arch == "osnet":
        from dlhub.vision.reid.osnet import build_osnet_reidentifier

        kwargs["variant"] = kwargs["variant"] or "osnet_tiny"
        return build_osnet_reidentifier(**kwargs)
    if arch == "agw":
        from dlhub.vision.reid.agw import build_agw_reidentifier

        kwargs["variant"] = kwargs["variant"] or "agw_tiny"
        return build_agw_reidentifier(**kwargs)
    if arch == "fastreid":
        from dlhub.vision.reid.fastreid import build_fastreid_reidentifier

        kwargs["variant"] = kwargs["variant"] or "fastreid_tiny"
        return build_fastreid_reidentifier(**kwargs)
    if arch == "transreid":
        from dlhub.vision.reid.transreid import build_transreid_reidentifier

        kwargs["variant"] = kwargs["variant"] or "transreid_tiny"
        return build_transreid_reidentifier(**kwargs)

    raise ValueError(f"Unknown arch: {arch_raw!r}")


def _batch_hard_triplet(embeddings: torch.Tensor, labels: torch.Tensor, margin: float = 0.2) -> torch.Tensor:
    distances = torch.cdist(embeddings, embeddings, p=2.0)
    total = embeddings.new_tensor(0.0)
    valid = 0
    for i in range(int(labels.shape[0])):
        same = labels == labels[i]
        same[i] = False
        diff = labels != labels[i]
        if not torch.any(same) or not torch.any(diff):
            continue
        hardest_pos = distances[i][same].max()
        hardest_neg = distances[i][diff].min()
        total = total + F.relu(hardest_pos - hardest_neg + float(margin))
        valid += 1
    if valid == 0:
        return embeddings.sum() * 0.0
    return total / float(valid)


def reid_loss(outputs: dict[str, torch.Tensor], labels: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    logits = outputs["logits"].to(torch.float32)
    embeddings = outputs["embedding"].to(torch.float32)
    targets = labels.to(torch.long)
    ce = F.cross_entropy(logits, targets)
    triplet = _batch_hard_triplet(embeddings, targets, margin=0.2)
    loss = ce + 0.5 * triplet
    return loss, {"ce": float(ce.detach().item()), "triplet": float(triplet.detach().item())}


def retrieval_top1_accuracy(embeddings: torch.Tensor, labels: torch.Tensor) -> float:
    if int(embeddings.shape[0]) < 2:
        return 0.0
    with torch.no_grad():
        scores = embeddings @ embeddings.t()
        scores.fill_diagonal_(-float("inf"))
        nn_index = scores.argmax(dim=1)
        return float((labels[nn_index] == labels).to(torch.float32).mean().item())


__all__ = [
    "ModelConfig",
    "build_model",
    "list_supported_arches",
    "reid_loss",
    "retrieval_top1_accuracy",
]

