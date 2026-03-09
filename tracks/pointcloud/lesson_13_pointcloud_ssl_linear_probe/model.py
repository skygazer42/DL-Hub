
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    ssl_arch: str = "simclr_pointnet:simclr_pointnet_small"
    ssl_dropout: float = 0.0
    in_channels: int = 3

    num_classes: int = 2
    freeze_ssl: bool = True


def list_supported_ssl_arches() -> list[str]:
    from dlhub.pointcloud.selfsupervised.barlowtwins import _VARIANTS as barlow
    from dlhub.pointcloud.selfsupervised.byol import _VARIANTS as byol
    from dlhub.pointcloud.selfsupervised.dino import _VARIANTS as dino
    from dlhub.pointcloud.selfsupervised.dinov2 import _VARIANTS as dinov2
    from dlhub.pointcloud.selfsupervised.ijepa import _VARIANTS as ijepa
    from dlhub.pointcloud.selfsupervised.simclr import _VARIANTS as simclr
    from dlhub.pointcloud.selfsupervised.swav import _VARIANTS as swav
    from dlhub.pointcloud.selfsupervised.vicreg import _VARIANTS as vicreg

    out: list[str] = []
    out += [f"simclr_pointnet:{k}" for k in sorted(simclr)]
    out += [f"byol_pointnet:{k}" for k in sorted(byol)]
    out += [f"dino_pointnet:{k}" for k in sorted(dino)]
    out += [f"dinov2_pointmae:{k}" for k in sorted(dinov2)]
    out += [f"ijepa_pointmae:{k}" for k in sorted(ijepa)]
    out += [f"swav_pointnet:{k}" for k in sorted(swav)]
    out += [f"barlowtwins_pointnet:{k}" for k in sorted(barlow)]
    out += [f"vicreg_pointnet:{k}" for k in sorted(vicreg)]
    return out


def _parse_arch(arch_raw: str, *, default_variant: str) -> tuple[str, str]:
    arch_raw = str(arch_raw).strip()
    if ":" in arch_raw:
        pref, name = arch_raw.split(":", 1)
        return pref.strip().lower(), name.strip()
    return arch_raw.lower(), str(default_variant)


def _build_ssl_model(*, ssl_arch: str, in_channels: int, dropout: float) -> nn.Module:
    pref, variant = _parse_arch(ssl_arch, default_variant="")

    if pref in {"simclr_pointnet", "simclr"}:
        from dlhub.pointcloud.selfsupervised.simclr import build_simclr_pointnet

        return build_simclr_pointnet(
            in_channels=int(in_channels),
            variant=str(variant) if variant else "simclr_pointnet_small",
            dropout=float(dropout),
        )

    if pref in {"byol_pointnet", "byol"}:
        from dlhub.pointcloud.selfsupervised.byol import build_byol_pointnet

        return build_byol_pointnet(
            in_channels=int(in_channels),
            variant=str(variant) if variant else "byol_pointnet_small",
            dropout=float(dropout),
        )

    if pref in {"dino_pointnet", "dino"}:
        from dlhub.pointcloud.selfsupervised.dino import build_dino_pointnet

        return build_dino_pointnet(
            in_channels=int(in_channels),
            variant=str(variant) if variant else "dino_pointnet_small",
            dropout=float(dropout),
        )

    if pref in {"dinov2_pointmae", "dinov2"}:
        from dlhub.pointcloud.selfsupervised.dinov2 import build_dinov2_pointmae

        return build_dinov2_pointmae(
            in_channels=int(in_channels),
            variant=str(variant) if variant else "dinov2_pointmae_small",
            dropout=float(dropout),
        )

    if pref in {"ijepa_pointmae", "ijepa"}:
        from dlhub.pointcloud.selfsupervised.ijepa import build_ijepa_pointmae

        return build_ijepa_pointmae(
            in_channels=int(in_channels),
            variant=str(variant) if variant else "ijepa_pointmae_small",
            dropout=float(dropout),
        )

    if pref in {"vicreg_pointnet", "vicreg"}:
        from dlhub.pointcloud.selfsupervised.vicreg import build_vicreg_pointnet

        return build_vicreg_pointnet(
            in_channels=int(in_channels),
            variant=str(variant) if variant else "vicreg_pointnet_small",
            dropout=float(dropout),
        )

    if pref in {"swav_pointnet", "swav"}:
        from dlhub.pointcloud.selfsupervised.swav import build_swav_pointnet

        return build_swav_pointnet(
            in_channels=int(in_channels),
            variant=str(variant) if variant else "swav_pointnet_small",
            dropout=float(dropout),
        )

    if pref in {"barlowtwins_pointnet", "barlowtwins", "barlow"}:
        from dlhub.pointcloud.selfsupervised.barlowtwins import build_barlowtwins_pointnet

        return build_barlowtwins_pointnet(
            in_channels=int(in_channels),
            variant=str(variant) if variant else "barlowtwins_pointnet_small",
            dropout=float(dropout),
        )

    raise ValueError(
        "Unknown ssl_arch: "
        f"{ssl_arch!r}. Supported prefixes: simclr_pointnet / byol_pointnet / dino_pointnet / dinov2_pointmae / ijepa_pointmae / swav_pointnet / barlowtwins_pointnet / vicreg_pointnet"
    )


class LinearProbeClassifier(nn.Module):
    """Linear probe (or fine-tune) on top of a self-supervised SSL model.

    We use the encoder features `h` from the SSL model (SimCLR/BYOL/VICReg).
    """

    def __init__(self, *, ssl_model: nn.Module, feature_dim: int, num_classes: int, freeze_ssl: bool) -> None:
        super().__init__()
        self.ssl = ssl_model
        self.freeze_ssl = bool(freeze_ssl)
        self.head = nn.Linear(int(feature_dim), int(num_classes))
        if self.freeze_ssl:
            self.ssl.requires_grad_(False)
            self.ssl.eval()

    def train(self, mode: bool = True):  # type: ignore[override]
        super().train(mode)
        if self.freeze_ssl:
            self.ssl.eval()
        return self

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        if self.freeze_ssl:
            with torch.no_grad():
                out = self.ssl(points)
                h = out["h"]
        else:
            out = self.ssl(points)
            h = out["h"]
        return self.head(h)


def build_model(cfg: ModelConfig) -> LinearProbeClassifier:
    ssl = _build_ssl_model(ssl_arch=str(cfg.ssl_arch), in_channels=int(cfg.in_channels), dropout=float(cfg.ssl_dropout))

    # Infer feature dim using a cheap forward on CPU.
    with torch.no_grad():
        dummy = torch.randn(2, 32, int(cfg.in_channels), dtype=torch.float32)
        out = ssl(dummy)
        if "h" not in out:
            raise ValueError("SSL model must return a dict with key 'h'")
        feat_dim = int(out["h"].shape[-1])

    return LinearProbeClassifier(
        ssl_model=ssl,
        feature_dim=feat_dim,
        num_classes=int(cfg.num_classes),
        freeze_ssl=bool(cfg.freeze_ssl),
    )


__all__ = [
    "LinearProbeClassifier",
    "ModelConfig",
    "build_model",
    "list_supported_ssl_arches",
]
