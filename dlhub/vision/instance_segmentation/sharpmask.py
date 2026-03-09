
import torch
from torch import nn
import torch.nn.functional as F

from dlhub.vision.backbones._blocks import ConvBNAct, scale_channels
from dlhub.vision.instance_segmentation._common import BackbonePyramid, InstanceTokenHead, check_nchw


class SharpMask(nn.Module):
    """SharpMask-style coarse-to-fine instance mask refiner."""

    def __init__(
        self,
        *,
        in_channels: int,
        stem_channels: int,
        p2_channels: int,
        p3_channels: int,
        p4_channels: int,
        hidden_channels: int,
        backbone_depth: int,
        num_proposals: int,
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
            act="relu",
        )
        self.tokens = InstanceTokenHead(int(p4_channels), int(hidden_channels), int(num_proposals), depth=2)
        self.score_head = nn.Linear(int(hidden_channels), 1)
        self.box_head = nn.Linear(int(hidden_channels), 4)
        self.coarse_head = nn.Linear(int(hidden_channels), int(mask_size) * int(mask_size))
        self.edge_head = nn.Sequential(
            ConvBNAct(int(p2_channels), int(hidden_channels), kernel_size=3, stride=1, act="relu"),
            nn.Conv2d(int(hidden_channels), 1, kernel_size=1),
        )
        self.mask_size = int(mask_size)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = check_nchw(x)
        p2, _, p4 = self.backbone(x)
        tokens = self.tokens(p4)
        b, k, _ = tokens.shape

        proposal_logits = self.score_head(tokens).squeeze(-1)
        proposal_boxes = torch.sigmoid(self.box_head(tokens))
        coarse_mask_logits = self.coarse_head(tokens).view(b, k, self.mask_size, self.mask_size)

        edge_logits = self.edge_head(p2)
        refined = F.interpolate(coarse_mask_logits, size=p2.shape[-2:], mode="bilinear", align_corners=False)
        mask_logits = refined + edge_logits.expand(-1, k, -1, -1)
        return {
            "proposal_logits": proposal_logits,
            "proposal_boxes": proposal_boxes,
            "coarse_mask_logits": coarse_mask_logits,
            "edge_logits": edge_logits,
            "mask_logits": mask_logits,
        }


_VARIANTS: dict[str, dict[str, int]] = {
    "sharpmask_tiny": {"stem": 24, "p2": 40, "p3": 64, "p4": 96, "hidden": 96, "depth": 1, "props": 16, "mask": 16},
    "sharpmask_small": {"stem": 24, "p2": 48, "p3": 80, "p4": 128, "hidden": 128, "depth": 2, "props": 24, "mask": 16},
    "sharpmask_base": {"stem": 32, "p2": 64, "p3": 96, "p4": 160, "hidden": 160, "depth": 3, "props": 32, "mask": 28},
}


def build_sharpmask_instance_segmenter(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "sharpmask_small",
    width_mult: float = 1.0,
    num_proposals: int | None = None,
) -> nn.Module:
    del num_classes
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown SharpMask variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]

    return SharpMask(
        in_channels=int(in_channels),
        stem_channels=scale_channels(int(spec["stem"]), float(width_mult), min_ch=16, divisor=8),
        p2_channels=scale_channels(int(spec["p2"]), float(width_mult), min_ch=16, divisor=8),
        p3_channels=scale_channels(int(spec["p3"]), float(width_mult), min_ch=16, divisor=8),
        p4_channels=scale_channels(int(spec["p4"]), float(width_mult), min_ch=16, divisor=8),
        hidden_channels=scale_channels(int(spec["hidden"]), float(width_mult), min_ch=16, divisor=8),
        backbone_depth=int(spec["depth"]),
        num_proposals=int(spec["props"]) if num_proposals is None else int(num_proposals),
        mask_size=int(spec["mask"]),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_sharpmask_instance_segmenter(in_channels=3, num_classes=3, variant="sharpmask_tiny", width_mult=0.5)
    out = m(x)
    print("sharpmask_tiny", {k: tuple(v.shape) for k, v in out.items()})
    loss = sum(v.mean() for v in out.values())
    loss.backward()
    print("ok")
