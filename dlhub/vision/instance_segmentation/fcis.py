import torch
import torch.nn.functional as F
from torch import nn

from dlhub.vision.backbones._blocks import ConvBNAct, scale_channels
from dlhub.vision.instance_segmentation._common import BackboneLowDet, check_nchw


class FCIS(nn.Module):
    """FCIS (Fully Convolutional Instance-aware Segmentation) style model (compact-first).

    Emits position-sensitive score maps (PSRoI) and mask maps.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        ps_size: int = 7,
        stem_channels: int = 24,
        low_channels: int = 48,
        det_channels: int = 96,
        backbone_depth: int = 2,
        head_channels: int = 96,
        head_convs: int = 2,
    ) -> None:
        super().__init__()
        nc = int(num_classes)
        if nc <= 0:
            raise ValueError("num_classes must be > 0")
        k = int(ps_size)
        if k <= 0:
            raise ValueError("ps_size must be > 0")

        self.backbone = BackboneLowDet(
            in_channels=int(in_channels),
            stem_channels=int(stem_channels),
            low_channels=int(low_channels),
            det_channels=int(det_channels),
            depth=int(backbone_depth),
            act="relu",
        )

        tower: list[nn.Module] = [
            ConvBNAct(int(det_channels), int(head_channels), kernel_size=3, stride=1, act="relu")
        ]
        for _ in range(int(head_convs) - 1):
            tower.append(
                ConvBNAct(
                    int(head_channels), int(head_channels), kernel_size=3, stride=1, act="relu"
                )
            )
        self.tower = nn.Sequential(*tower)

        self.ps_scores = nn.Conv2d(int(head_channels), nc * k * k, kernel_size=1, bias=True)
        self.ps_masks = nn.Conv2d(int(head_channels), k * k, kernel_size=1, bias=True)
        self.ps_size = k
        self.num_classes = nc

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = check_nchw(x)
        _, det = self.backbone(x)
        t = self.tower(det)
        ps_scores = self.ps_scores(t)
        ps_masks = self.ps_masks(t)
        # A crude "assembled" mask map at /4 for visualization/training hooks.
        mask_logits = F.interpolate(ps_masks, scale_factor=2, mode="nearest")
        return {"ps_scores": ps_scores, "ps_masks": ps_masks, "mask_logits": mask_logits}


_VARIANTS: dict[str, dict] = {
    "fcis_tiny": {"stem": 24, "low": 40, "det": 80, "depth": 1, "head": 80, "ps": 5},
    "fcis_small": {"stem": 24, "low": 48, "det": 96, "depth": 2, "head": 96, "ps": 7},
    "fcis_base": {"stem": 32, "low": 64, "det": 128, "depth": 3, "head": 128, "ps": 7},
}


def build_fcis_instance_segmenter(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "fcis_small",
    width_mult: float = 1.0,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown FCIS variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]

    stem = scale_channels(int(spec["stem"]), float(width_mult), min_ch=16, divisor=8)
    low = scale_channels(int(spec["low"]), float(width_mult), min_ch=16, divisor=8)
    det = scale_channels(int(spec["det"]), float(width_mult), min_ch=16, divisor=8)
    head = scale_channels(int(spec["head"]), float(width_mult), min_ch=16, divisor=8)

    return FCIS(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        ps_size=int(spec["ps"]),
        stem_channels=int(stem),
        low_channels=int(low),
        det_channels=int(det),
        backbone_depth=int(spec["depth"]),
        head_channels=int(head),
        head_convs=2,
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_fcis_instance_segmenter(
        in_channels=3, num_classes=3, variant="fcis_tiny", width_mult=0.5
    )
    out = m(x)
    print("fcis_tiny", {k: tuple(v.shape) for k, v in out.items()})
    loss = sum(v.mean() for v in out.values())
    loss.backward()
    print("ok")
