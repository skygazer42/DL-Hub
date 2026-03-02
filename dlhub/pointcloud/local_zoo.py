from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable

from torch import nn

from .backbones import (
    build_deepsets_classifier,
    build_dgcnn_classifier,
    build_curvenet_classifier,
    build_gdanet_classifier,
    build_pointgat_classifier,
    build_pointgcn_classifier,
    build_pointweb_classifier,
    build_kpconv_classifier,
    build_asnl_classifier,
    build_paconv_classifier,
    build_pct_classifier,
    build_pointbert_classifier,
    build_point2seq_classifier,
    build_pointsift_classifier,
    build_pointmae_classifier,
    build_point_transformer_classifier,
    build_pointcnn_classifier,
    build_pointconv_classifier,
    build_pvcnn_classifier,
    build_randlanet_classifier,
    build_rscnn_classifier,
    build_simpleview_classifier,
    build_spidercnn_classifier,
    build_shellnet_classifier,
    build_pointmixer_classifier,
    build_pointmlp_classifier,
    build_pointnet2_classifier,
    build_pointnet_classifier,
    build_pointnext_classifier,
)


class UnknownLocalArch(KeyError):
    pass


@dataclass(frozen=True)
class BuildConfig:
    in_channels: int
    num_classes: int
    num_points: int
    width_mult: float
    dropout: float


Builder = Callable[[BuildConfig], nn.Module]


def _split_arch_id(arch_id: str) -> tuple[str, str]:
    arch_id = str(arch_id).strip()
    if ":" not in arch_id:
        raise ValueError(f"Expected a namespaced arch id like 'pc:pointnet', got: {arch_id!r}")
    prefix, name = arch_id.split(":", 1)
    prefix = prefix.strip().lower()
    name = name.strip()
    if not prefix or not name:
        raise ValueError(f"Invalid arch id: {arch_id!r}")
    return prefix, name


def _registry() -> dict[str, Builder]:
    r: dict[str, Builder] = {}

    # Set-based families
    r["pointnet"] = lambda cfg: build_pointnet_classifier(
        in_channels=cfg.in_channels,
        num_classes=cfg.num_classes,
        width_mult=cfg.width_mult,
        dropout=cfg.dropout,
        variant="pointnet",
    )
    r["pointnet_tnet"] = lambda cfg: build_pointnet_classifier(
        in_channels=cfg.in_channels,
        num_classes=cfg.num_classes,
        width_mult=cfg.width_mult,
        dropout=cfg.dropout,
        variant="pointnet_tnet",
    )
    r["deepsets"] = lambda cfg: build_deepsets_classifier(
        in_channels=cfg.in_channels,
        num_classes=cfg.num_classes,
        width_mult=cfg.width_mult,
        dropout=cfg.dropout,
        variant="deepsets",
    )
    r["deepsets_mean"] = lambda cfg: build_deepsets_classifier(
        in_channels=cfg.in_channels,
        num_classes=cfg.num_classes,
        width_mult=cfg.width_mult,
        dropout=cfg.dropout,
        variant="deepsets_mean",
    )

    for v in ["pointnet2_ssg", "pointnet2_msg"]:
        r[v] = lambda cfg, v=v: build_pointnet2_classifier(
            in_channels=cfg.in_channels,
            num_classes=cfg.num_classes,
            num_points=cfg.num_points,
            width_mult=cfg.width_mult,
            dropout=cfg.dropout,
            variant=v,
        )

    # Graph families
    r["dgcnn"] = lambda cfg: build_dgcnn_classifier(
        in_channels=cfg.in_channels,
        num_classes=cfg.num_classes,
        num_points=cfg.num_points,
        width_mult=cfg.width_mult,
        dropout=cfg.dropout,
        variant="dgcnn",
    )
    r["dgcnn_static"] = lambda cfg: build_dgcnn_classifier(
        in_channels=cfg.in_channels,
        num_classes=cfg.num_classes,
        num_points=cfg.num_points,
        width_mult=cfg.width_mult,
        dropout=cfg.dropout,
        variant="dgcnn_static",
    )

    for v in ["pointgcn", "pointgcn_small"]:
        r[v] = lambda cfg, v=v: build_pointgcn_classifier(
            in_channels=cfg.in_channels,
            num_classes=cfg.num_classes,
            num_points=cfg.num_points,
            width_mult=cfg.width_mult,
            dropout=cfg.dropout,
            variant=v,
        )
    for v in ["pointgat", "pointgat_small"]:
        r[v] = lambda cfg, v=v: build_pointgat_classifier(
            in_channels=cfg.in_channels,
            num_classes=cfg.num_classes,
            num_points=cfg.num_points,
            width_mult=cfg.width_mult,
            dropout=cfg.dropout,
            variant=v,
        )
    for v in ["pointweb", "pointweb_small"]:
        r[v] = lambda cfg, v=v: build_pointweb_classifier(
            in_channels=cfg.in_channels,
            num_classes=cfg.num_classes,
            num_points=cfg.num_points,
            width_mult=cfg.width_mult,
            dropout=cfg.dropout,
            variant=v,
        )

    # Transformer-ish
    for v in ["point_transformer", "point_transformer_tiny", "point_transformer_small"]:
        r[v] = lambda cfg, v=v: build_point_transformer_classifier(
            in_channels=cfg.in_channels,
            num_classes=cfg.num_classes,
            num_points=cfg.num_points,
            width_mult=cfg.width_mult,
            dropout=cfg.dropout,
            variant=v,
        )
    for v in ["pct", "pct_small", "pct_base"]:
        r[v] = lambda cfg, v=v: build_pct_classifier(
            in_channels=cfg.in_channels,
            num_classes=cfg.num_classes,
            num_points=cfg.num_points,
            width_mult=cfg.width_mult,
            dropout=cfg.dropout,
            variant=v,
        )

    for v in ["pointbert", "pointbert_small"]:
        r[v] = lambda cfg, v=v: build_pointbert_classifier(
            in_channels=cfg.in_channels,
            num_classes=cfg.num_classes,
            num_points=cfg.num_points,
            width_mult=cfg.width_mult,
            dropout=cfg.dropout,
            variant=v,
        )
    for v in ["pointmae", "pointmae_small"]:
        r[v] = lambda cfg, v=v: build_pointmae_classifier(
            in_channels=cfg.in_channels,
            num_classes=cfg.num_classes,
            num_points=cfg.num_points,
            width_mult=cfg.width_mult,
            dropout=cfg.dropout,
            variant=v,
        )

    # MLP-ish
    for v in ["pointmlp", "pointmlp_small", "pointmlp_base"]:
        r[v] = lambda cfg, v=v: build_pointmlp_classifier(
            in_channels=cfg.in_channels,
            num_classes=cfg.num_classes,
            num_points=cfg.num_points,
            width_mult=cfg.width_mult,
            dropout=cfg.dropout,
            variant=v,
        )
    for v in ["pointnext_tiny", "pointnext_small", "pointnext_base"]:
        r[v] = lambda cfg, v=v: build_pointnext_classifier(
            in_channels=cfg.in_channels,
            num_classes=cfg.num_classes,
            num_points=cfg.num_points,
            width_mult=cfg.width_mult,
            dropout=cfg.dropout,
            variant=v,
        )
    for v in ["pointmixer", "pointmixer_small"]:
        r[v] = lambda cfg, v=v: build_pointmixer_classifier(
            in_channels=cfg.in_channels,
            num_classes=cfg.num_classes,
            num_points=cfg.num_points,
            width_mult=cfg.width_mult,
            dropout=cfg.dropout,
            variant=v,
        )

    # Conv-ish
    for v in ["pointcnn", "pointcnn_small", "pointcnn_base"]:
        r[v] = lambda cfg, v=v: build_pointcnn_classifier(
            in_channels=cfg.in_channels,
            num_classes=cfg.num_classes,
            num_points=cfg.num_points,
            width_mult=cfg.width_mult,
            dropout=cfg.dropout,
            variant=v,
        )
    for v in ["pointconv", "pointconv_small", "pointconv_base"]:
        r[v] = lambda cfg, v=v: build_pointconv_classifier(
            in_channels=cfg.in_channels,
            num_classes=cfg.num_classes,
            num_points=cfg.num_points,
            width_mult=cfg.width_mult,
            dropout=cfg.dropout,
            variant=v,
        )
    for v in ["kpconv", "kpconv_small", "kpconv_base"]:
        r[v] = lambda cfg, v=v: build_kpconv_classifier(
            in_channels=cfg.in_channels,
            num_classes=cfg.num_classes,
            num_points=cfg.num_points,
            width_mult=cfg.width_mult,
            dropout=cfg.dropout,
            variant=v,
        )

    # Extra mainstream architectures (simplified)
    for v in ["spidercnn", "spidercnn_small"]:
        r[v] = lambda cfg, v=v: build_spidercnn_classifier(
            in_channels=cfg.in_channels,
            num_classes=cfg.num_classes,
            num_points=cfg.num_points,
            width_mult=cfg.width_mult,
            dropout=cfg.dropout,
            variant=v,
        )
    for v in ["rscnn", "rscnn_small"]:
        r[v] = lambda cfg, v=v: build_rscnn_classifier(
            in_channels=cfg.in_channels,
            num_classes=cfg.num_classes,
            num_points=cfg.num_points,
            width_mult=cfg.width_mult,
            dropout=cfg.dropout,
            variant=v,
        )
    for v in ["paconv", "paconv_small"]:
        r[v] = lambda cfg, v=v: build_paconv_classifier(
            in_channels=cfg.in_channels,
            num_classes=cfg.num_classes,
            num_points=cfg.num_points,
            width_mult=cfg.width_mult,
            dropout=cfg.dropout,
            variant=v,
        )
    r["curvenet"] = lambda cfg: build_curvenet_classifier(
        in_channels=cfg.in_channels,
        num_classes=cfg.num_classes,
        num_points=cfg.num_points,
        width_mult=cfg.width_mult,
        dropout=cfg.dropout,
        variant="curvenet",
    )
    r["gdanet"] = lambda cfg: build_gdanet_classifier(
        in_channels=cfg.in_channels,
        num_classes=cfg.num_classes,
        num_points=cfg.num_points,
        width_mult=cfg.width_mult,
        dropout=cfg.dropout,
        variant="gdanet",
    )
    r["pointsift"] = lambda cfg: build_pointsift_classifier(
        in_channels=cfg.in_channels,
        num_classes=cfg.num_classes,
        num_points=cfg.num_points,
        width_mult=cfg.width_mult,
        dropout=cfg.dropout,
        variant="pointsift",
    )
    for v in ["point2seq", "point2seq_small"]:
        r[v] = lambda cfg, v=v: build_point2seq_classifier(
            in_channels=cfg.in_channels,
            num_classes=cfg.num_classes,
            num_points=cfg.num_points,
            width_mult=cfg.width_mult,
            dropout=cfg.dropout,
            variant=v,
        )
    r["asnl"] = lambda cfg: build_asnl_classifier(
        in_channels=cfg.in_channels,
        num_classes=cfg.num_classes,
        num_points=cfg.num_points,
        width_mult=cfg.width_mult,
        dropout=cfg.dropout,
        variant="asnl",
    )
    r["randlanet"] = lambda cfg: build_randlanet_classifier(
        in_channels=cfg.in_channels,
        num_classes=cfg.num_classes,
        num_points=cfg.num_points,
        width_mult=cfg.width_mult,
        dropout=cfg.dropout,
        variant="randlanet",
    )
    for v in ["pvcnn", "pvcnn_small"]:
        r[v] = lambda cfg, v=v: build_pvcnn_classifier(
            in_channels=cfg.in_channels,
            num_classes=cfg.num_classes,
            num_points=cfg.num_points,
            width_mult=cfg.width_mult,
            dropout=cfg.dropout,
            variant=v,
        )
    r["simpleview"] = lambda cfg: build_simpleview_classifier(
        in_channels=cfg.in_channels,
        num_classes=cfg.num_classes,
        num_points=cfg.num_points,
        width_mult=cfg.width_mult,
        dropout=cfg.dropout,
        variant="simpleview",
    )

    for v in ["shellnet", "shellnet_small", "shellnet_base"]:
        r[v] = lambda cfg, v=v: build_shellnet_classifier(
            in_channels=cfg.in_channels,
            num_classes=cfg.num_classes,
            num_points=cfg.num_points,
            width_mult=cfg.width_mult,
            dropout=cfg.dropout,
            variant=v,
        )

    # Friendly aliases
    r["pointnet2"] = r["pointnet2_ssg"]
    r["pt"] = r["point_transformer"]
    r["pc_transformer"] = r["pct"]
    r["pointnext"] = r["pointnext_tiny"]

    return r


_REGISTRY = _registry()


def list_local_arches() -> list[str]:
    return [f"pc:{name}" for name in sorted(_REGISTRY)]


def build_local_model(
    arch_id: str,
    *,
    in_channels: int,
    num_classes: int,
    num_points: int,
    width_mult: float = 1.0,
    dropout: float = 0.1,
) -> nn.Module:
    prefix, name = _split_arch_id(arch_id)
    if prefix not in {"pc", "local"}:
        raise ValueError(f"Unsupported pointcloud prefix: {prefix!r} (arch_id={arch_id!r})")

    builder = _REGISTRY.get(name)
    if builder is None:
        raise UnknownLocalArch(
            f"Unknown pointcloud arch: {arch_id!r}. Tip: see `list_local_arches()` or `python scripts/pointcloud_zoo.py --list`."
        )
    return builder(
        BuildConfig(
            in_channels=int(in_channels),
            num_classes=int(num_classes),
            num_points=int(num_points),
            width_mult=float(width_mult),
            dropout=float(dropout),
        )
    )


__all__ = [
    "BuildConfig",
    "UnknownLocalArch",
    "build_local_model",
    "list_local_arches",
]
