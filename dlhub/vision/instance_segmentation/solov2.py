from __future__ import annotations

import torch
from torch import nn

from dlhub.vision.backbones._blocks import ConvBNAct, scale_channels
from dlhub.vision.instance_segmentation._common import BackboneLowDet, check_nchw


class SOLOv2(nn.Module):
    """SOLOv2-style instance segmentation (toy-first).

    Produces per-cell category logits and dynamic mask kernels applied to a mask feature map.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        mask_channels: int = 32,
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
        mc = int(mask_channels)
        if mc <= 0:
            raise ValueError("mask_channels must be > 0")

        self.backbone = BackboneLowDet(
            in_channels=int(in_channels),
            stem_channels=int(stem_channels),
            low_channels=int(low_channels),
            det_channels=int(det_channels),
            depth=int(backbone_depth),
            act="relu",
        )

        self.mask_feat = nn.Sequential(
            ConvBNAct(int(low_channels), mc, kernel_size=3, stride=1, act="relu"),
            ConvBNAct(mc, mc, kernel_size=3, stride=1, act="relu"),
        )

        tower: list[nn.Module] = [ConvBNAct(int(det_channels), int(head_channels), kernel_size=3, stride=1, act="relu")]
        for _ in range(int(head_convs) - 1):
            tower.append(ConvBNAct(int(head_channels), int(head_channels), kernel_size=3, stride=1, act="relu"))
        self.tower = nn.Sequential(*tower)

        self.cat = nn.Conv2d(int(head_channels), nc, kernel_size=3, padding=1)
        self.kernel = nn.Conv2d(int(head_channels), mc, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = check_nchw(x)
        low, det = self.backbone(x)

        mf = self.mask_feat(low)  # (B,M,H/4,W/4)
        t = self.tower(det)
        cat_logits = self.cat(t)  # (B,C,H/8,W/8)
        kernels = self.kernel(t)  # (B,M,H/8,W/8)

        b, m, h4, w4 = mf.shape
        slots = kernels.shape[-2] * kernels.shape[-1]
        ker_flat = kernels.permute(0, 2, 3, 1).reshape(b, slots, m)  # (B,S,M)
        mf_flat = mf.reshape(b, m, h4 * w4)  # (B,M,HW)
        mask_flat = torch.bmm(ker_flat, mf_flat)  # (B,S,HW)
        mask_logits = mask_flat.view(b, slots, h4, w4)
        return {"cat_logits": cat_logits, "mask_logits": mask_logits, "mask_feat": mf, "mask_kernels": kernels}


_VARIANTS: dict[str, dict] = {
    "solov2_tiny": {"stem": 24, "low": 40, "det": 80, "depth": 1, "head": 80, "mask": 24},
    "solov2_small": {"stem": 24, "low": 48, "det": 96, "depth": 2, "head": 96, "mask": 32},
    "solov2_base": {"stem": 32, "low": 64, "det": 128, "depth": 3, "head": 128, "mask": 48},
}


def build_solov2_instance_segmenter(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "solov2_small",
    width_mult: float = 1.0,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown SOLOv2 variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    stem = scale_channels(int(spec["stem"]), float(width_mult), min_ch=16, divisor=8)
    low = scale_channels(int(spec["low"]), float(width_mult), min_ch=16, divisor=8)
    det = scale_channels(int(spec["det"]), float(width_mult), min_ch=16, divisor=8)
    head = scale_channels(int(spec["head"]), float(width_mult), min_ch=16, divisor=8)
    mask = scale_channels(int(spec["mask"]), float(width_mult), min_ch=16, divisor=8)
    return SOLOv2(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        mask_channels=int(mask),
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
    m = build_solov2_instance_segmenter(in_channels=3, num_classes=3, variant="solov2_tiny", width_mult=0.5)
    out = m(x)
    print("solov2_tiny", {k: tuple(v.shape) for k, v in out.items()})
    loss = sum(v.mean() for v in out.values())
    loss.backward()
    print("ok")

