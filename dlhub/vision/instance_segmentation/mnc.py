from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from dlhub.vision.backbones._blocks import scale_channels
from dlhub.vision.instance_segmentation._common import BackbonePyramid, InstanceTokenHead, check_nchw


class MNC(nn.Module):
    """Multi-task Network Cascade style instance segmenter."""

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
        num_instances: int,
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
        self.tokens = InstanceTokenHead(int(p4_channels), int(hidden_channels), int(num_instances), depth=3)
        self.stage1_head = nn.Linear(int(hidden_channels), 1 + 4)
        self.stage2_mask = nn.Linear(int(hidden_channels), int(mask_size) * int(mask_size))
        self.cls_head = nn.Linear(int(hidden_channels), int(num_classes))
        self.refine_head = nn.Linear(int(hidden_channels), 4)
        self.mask_size = int(mask_size)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = check_nchw(x)
        p2, _, p4 = self.backbone(x)
        tokens = self.tokens(p4)
        b, k, _ = tokens.shape

        stage1 = self.stage1_head(tokens)
        proposal_logits = stage1[..., 0]
        proposal_boxes = torch.sigmoid(stage1[..., 1:])
        stage1_mask_logits = self.stage2_mask(tokens).view(b, k, self.mask_size, self.mask_size)
        stage2_mask_logits = F.interpolate(stage1_mask_logits, size=p2.shape[-2:], mode="bilinear", align_corners=False)
        stage2_mask_logits = stage2_mask_logits + p2.mean(dim=1, keepdim=True).expand(-1, k, -1, -1)

        cls_logits = self.cls_head(tokens)
        refined_boxes = torch.sigmoid(proposal_boxes + 0.25 * self.refine_head(tokens))
        return {
            "proposal_logits": proposal_logits,
            "proposal_boxes": proposal_boxes,
            "cls_logits": cls_logits,
            "refined_boxes": refined_boxes,
            "stage1_mask_logits": stage1_mask_logits,
            "stage2_mask_logits": stage2_mask_logits,
        }


_VARIANTS: dict[str, dict[str, int]] = {
    "mnc_tiny": {"stem": 24, "p2": 40, "p3": 64, "p4": 96, "hidden": 96, "depth": 1, "instances": 16, "mask": 16},
    "mnc_small": {"stem": 24, "p2": 48, "p3": 80, "p4": 128, "hidden": 128, "depth": 2, "instances": 24, "mask": 16},
    "mnc_base": {"stem": 32, "p2": 64, "p3": 96, "p4": 160, "hidden": 160, "depth": 3, "instances": 32, "mask": 28},
}


def build_mnc_instance_segmenter(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "mnc_small",
    width_mult: float = 1.0,
    num_instances: int | None = None,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown MNC variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]

    return MNC(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        stem_channels=scale_channels(int(spec["stem"]), float(width_mult), min_ch=16, divisor=8),
        p2_channels=scale_channels(int(spec["p2"]), float(width_mult), min_ch=16, divisor=8),
        p3_channels=scale_channels(int(spec["p3"]), float(width_mult), min_ch=16, divisor=8),
        p4_channels=scale_channels(int(spec["p4"]), float(width_mult), min_ch=16, divisor=8),
        hidden_channels=scale_channels(int(spec["hidden"]), float(width_mult), min_ch=16, divisor=8),
        backbone_depth=int(spec["depth"]),
        num_instances=int(spec["instances"]) if num_instances is None else int(num_instances),
        mask_size=int(spec["mask"]),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_mnc_instance_segmenter(in_channels=3, num_classes=3, variant="mnc_tiny", width_mult=0.5)
    out = m(x)
    print("mnc_tiny", {k: tuple(v.shape) for k, v in out.items()})
    loss = sum(v.mean() for v in out.values())
    loss.backward()
    print("ok")
