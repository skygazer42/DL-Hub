from dataclasses import dataclass

from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    arch: str = "pointnet"  # pointnet | dgcnn
    in_channels: int = 3
    num_classes: int = 2
    num_points: int = 256

    # Shared knobs
    hidden_features: int = 64
    dropout: float = 0.1

    # DGCNN-only
    k: int = 10
    dynamic_graph: bool = True


def list_supported_arches() -> list[str]:
    return ["pointnet", "dgcnn", "dgcnn_static"]


def build_model(cfg: ModelConfig) -> nn.Module:
    arch_raw = str(cfg.arch).strip()
    arch = arch_raw.lower()

    if arch in {"pointnet", "pointnet_partseg"}:
        from tracks.pointcloud.lesson_05_pointnet_compact_partseg.model import ModelConfig as PNConfig
        from tracks.pointcloud.lesson_05_pointnet_compact_partseg.model import PointNetPartSeg

        return PointNetPartSeg(
            PNConfig(
                in_channels=int(cfg.in_channels),
                hidden_features=int(cfg.hidden_features),
                num_classes=int(cfg.num_classes),
                dropout=float(cfg.dropout),
            )
        )

    if arch in {"dgcnn", "dgcnn_partseg", "dgcnn_static"}:
        from tracks.pointcloud.lesson_06_dgcnn_compact_partseg.model import DGCNNPartSeg
        from tracks.pointcloud.lesson_06_dgcnn_compact_partseg.model import ModelConfig as DGCNNConfig

        dynamic = bool(cfg.dynamic_graph)
        if arch == "dgcnn_static":
            dynamic = False
        return DGCNNPartSeg(
            DGCNNConfig(
                k=int(cfg.k),
                hidden_features=int(cfg.hidden_features),
                dropout=float(cfg.dropout),
                num_classes=int(cfg.num_classes),
                dynamic_graph=bool(dynamic),
            )
        )

    raise ValueError(f"Unknown arch: {arch_raw!r}. Supported: {list_supported_arches()}")


__all__ = ["ModelConfig", "build_model", "list_supported_arches"]
