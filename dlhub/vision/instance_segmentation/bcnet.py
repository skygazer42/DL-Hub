import torch
import torch.nn.functional as F
from torch import nn

from dlhub.vision.backbones._blocks import ConvBNAct, scale_channels
from dlhub.vision.instance_segmentation._common import (
    BackbonePyramid,
    InstanceTokenHead,
    check_nchw,
)


class BCNet(nn.Module):
    """Boundary-aware collaborative network for instance masks (compact-first)."""

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
        self.boundary_head = nn.Sequential(
            ConvBNAct(int(p2_channels), int(hidden_channels), kernel_size=3, stride=1, act="relu"),
            nn.Conv2d(int(hidden_channels), 1, kernel_size=1),
        )
        self.tokens = InstanceTokenHead(
            int(p4_channels), int(hidden_channels), int(num_instances), depth=2
        )
        self.cls_head = nn.Linear(int(hidden_channels), int(num_classes))
        self.box_head = nn.Linear(int(hidden_channels), 4)
        self.mask_head = nn.Linear(int(hidden_channels), int(mask_size) * int(mask_size))
        self.score_head = nn.Linear(int(hidden_channels), 1)
        self.mask_size = int(mask_size)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = check_nchw(x)
        p2, _, p4 = self.backbone(x)
        tokens = self.tokens(p4)
        b, k, _ = tokens.shape

        boundary_logits = self.boundary_head(p2)
        proposal_logits = self.score_head(tokens).squeeze(-1)
        cls_logits = self.cls_head(tokens)
        proposal_boxes = torch.sigmoid(self.box_head(tokens))
        mask_logits = self.mask_head(tokens).view(b, k, self.mask_size, self.mask_size)
        mask_logits = F.interpolate(
            mask_logits, size=p2.shape[-2:], mode="bilinear", align_corners=False
        )
        mask_logits = mask_logits + boundary_logits.expand(-1, k, -1, -1)
        return {
            "proposal_logits": proposal_logits,
            "cls_logits": cls_logits,
            "proposal_boxes": proposal_boxes,
            "boundary_logits": boundary_logits,
            "mask_logits": mask_logits,
        }


_VARIANTS: dict[str, dict[str, int]] = {
    "bcnet_tiny": {
        "stem": 24,
        "p2": 40,
        "p3": 64,
        "p4": 96,
        "hidden": 96,
        "depth": 1,
        "instances": 16,
        "mask": 16,
    },
    "bcnet_small": {
        "stem": 24,
        "p2": 48,
        "p3": 80,
        "p4": 128,
        "hidden": 128,
        "depth": 2,
        "instances": 24,
        "mask": 16,
    },
    "bcnet_base": {
        "stem": 32,
        "p2": 64,
        "p3": 96,
        "p4": 160,
        "hidden": 160,
        "depth": 3,
        "instances": 32,
        "mask": 28,
    },
}


def build_bcnet_instance_segmenter(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "bcnet_small",
    width_mult: float = 1.0,
    num_instances: int | None = None,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown BCNet variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]

    return BCNet(
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
        num_instances=int(spec["instances"]) if num_instances is None else int(num_instances),
        mask_size=int(spec["mask"]),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_bcnet_instance_segmenter(
        in_channels=3, num_classes=3, variant="bcnet_tiny", width_mult=0.5
    )
    out = m(x)
    print("bcnet_tiny", {k: tuple(v.shape) for k, v in out.items()})
    loss = sum(v.mean() for v in out.values())
    loss.backward()
    print("ok")
