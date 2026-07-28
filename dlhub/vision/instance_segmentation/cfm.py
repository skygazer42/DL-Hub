import torch
import torch.nn.functional as F
from torch import nn

from dlhub.vision.backbones._blocks import scale_channels
from dlhub.vision.instance_segmentation._common import (
    BackbonePyramid,
    InstanceTokenHead,
    check_nchw,
)


class CFM(nn.Module):
    """Convolutional Feature Masking style instance segmenter (compact-first)."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        stem_channels: int,
        p2_channels: int,
        p3_channels: int,
        p4_channels: int,
        hidden_channels: int,
        backbone_depth: int,
        num_regions: int,
        mask_size: int,
    ) -> None:
        super().__init__()
        self.backbone = BackbonePyramid(
            in_channels=int(in_channels),
            stem_channels=int(stem_channels),
            p2_channels=int(p2_channels),
            p3_channels=int(p3_channels),
            p4_channels=int(p4_channels),
            depth=int(backbone_depth),
        )
        self.tokens = InstanceTokenHead(
            int(p4_channels), int(hidden_channels), int(num_regions), depth=2
        )
        self.region_head = nn.Linear(int(hidden_channels), 1)
        self.cls_head = nn.Linear(int(hidden_channels), int(num_classes))
        self.box_head = nn.Linear(int(hidden_channels), 4)
        self.mask_head = nn.Linear(int(hidden_channels), int(mask_size) * int(mask_size))
        self.mask_size = int(mask_size)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = check_nchw(x)
        p2, p3, p4 = self.backbone(x)
        tokens = self.tokens(p4)
        b, k, _ = tokens.shape

        proposal_logits = self.region_head(tokens).squeeze(-1)
        cls_logits = self.cls_head(tokens)
        proposal_boxes = torch.sigmoid(self.box_head(tokens))
        coarse_mask_logits = self.mask_head(tokens).view(b, k, self.mask_size, self.mask_size)

        seg_context = (
            p2.mean(dim=1, keepdim=True)
            + F.interpolate(p3.mean(dim=1, keepdim=True), size=p2.shape[-2:], mode="nearest")
        ) * 0.5
        mask_logits = F.interpolate(
            coarse_mask_logits, size=p2.shape[-2:], mode="bilinear", align_corners=False
        )
        mask_logits = mask_logits + seg_context.expand(-1, k, -1, -1)
        return {
            "proposal_logits": proposal_logits,
            "cls_logits": cls_logits,
            "proposal_boxes": proposal_boxes,
            "coarse_mask_logits": coarse_mask_logits,
            "mask_logits": mask_logits,
        }


_VARIANTS: dict[str, dict[str, int]] = {
    "cfm_tiny": {
        "stem": 24,
        "p2": 40,
        "p3": 64,
        "p4": 96,
        "hidden": 96,
        "depth": 1,
        "regions": 16,
        "mask": 16,
    },
    "cfm_small": {
        "stem": 24,
        "p2": 48,
        "p3": 80,
        "p4": 128,
        "hidden": 128,
        "depth": 2,
        "regions": 24,
        "mask": 16,
    },
    "cfm_base": {
        "stem": 32,
        "p2": 64,
        "p3": 96,
        "p4": 160,
        "hidden": 160,
        "depth": 3,
        "regions": 32,
        "mask": 28,
    },
}


def build_cfm_instance_segmenter(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "cfm_small",
    width_mult: float = 1.0,
    num_regions: int | None = None,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown CFM variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]

    return CFM(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        stem_channels=scale_channels(int(spec["stem"]), float(width_mult), min_ch=16, divisor=8),
        p2_channels=scale_channels(int(spec["p2"]), float(width_mult), min_ch=16, divisor=8),
        p3_channels=scale_channels(int(spec["p3"]), float(width_mult), min_ch=16, divisor=8),
        p4_channels=scale_channels(int(spec["p4"]), float(width_mult), min_ch=16, divisor=8),
        hidden_channels=scale_channels(
            int(spec["hidden"]), float(width_mult), min_ch=16, divisor=8
        ),
        backbone_depth=int(spec["depth"]),
        num_regions=int(spec["regions"]) if num_regions is None else int(num_regions),
        mask_size=int(spec["mask"]),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_cfm_instance_segmenter(
        in_channels=3, num_classes=3, variant="cfm_tiny", width_mult=0.5
    )
    out = m(x)
    print("cfm_tiny", {k: tuple(v.shape) for k, v in out.items()})
    loss = sum(v.mean() for v in out.values())
    loss.backward()
    print("ok")
