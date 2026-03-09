
import torch
from torch import nn
import torch.nn.functional as F

from dlhub.vision.backbones._blocks import scale_channels
from dlhub.vision.instance_segmentation._common import BackbonePyramid, InstanceTokenHead, check_nchw


class MaskDINO(nn.Module):
    """Mask DINO-style denoising query instance segmenter."""

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
        num_queries: int,
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
        self.tokens = InstanceTokenHead(int(p4_channels), int(hidden_channels), int(num_queries), depth=3)
        self.cls_head = nn.Linear(int(hidden_channels), int(num_classes))
        self.box_head = nn.Linear(int(hidden_channels), 4)
        self.mask_head = nn.Linear(int(hidden_channels), int(mask_size) * int(mask_size))
        self.ref_head = nn.Linear(int(hidden_channels), 2)
        self.dn_proj = nn.Linear(int(p2_channels), int(hidden_channels))
        self.mask_size = int(mask_size)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = check_nchw(x)
        p2, p3, p4 = self.backbone(x)
        tokens = self.tokens(p4)
        b, q, _ = tokens.shape

        cls_logits = self.cls_head(tokens)
        boxes = torch.sigmoid(self.box_head(tokens))
        ref_points = torch.sigmoid(self.ref_head(tokens))
        mask_logits = self.mask_head(tokens).view(b, q, self.mask_size, self.mask_size)
        mask_logits = F.interpolate(mask_logits, size=p2.shape[-2:], mode="bilinear", align_corners=False)
        mask_logits = mask_logits + 0.5 * F.interpolate(p3.mean(dim=1, keepdim=True), size=p2.shape[-2:], mode="nearest").expand(-1, q, -1, -1)

        dn_context = self.dn_proj(p2.mean(dim=(-2, -1))).unsqueeze(1)
        dn_tokens = tokens + dn_context
        dn_cls_logits = self.cls_head(dn_tokens)
        dn_mask_logits = self.mask_head(dn_tokens).view(b, q, self.mask_size, self.mask_size)
        return {
            "cls_logits": cls_logits,
            "boxes": boxes,
            "ref_points": ref_points,
            "mask_logits": mask_logits,
            "dn_cls_logits": dn_cls_logits,
            "dn_mask_logits": dn_mask_logits,
        }


_VARIANTS: dict[str, dict[str, int]] = {
    "mask_dino_tiny": {"stem": 24, "p2": 40, "p3": 64, "p4": 96, "hidden": 96, "depth": 1, "queries": 16, "mask": 16},
    "mask_dino_small": {"stem": 24, "p2": 48, "p3": 80, "p4": 128, "hidden": 128, "depth": 2, "queries": 24, "mask": 16},
    "mask_dino_base": {"stem": 32, "p2": 64, "p3": 96, "p4": 160, "hidden": 160, "depth": 3, "queries": 32, "mask": 28},
}


def build_mask_dino_instance_segmenter(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "mask_dino_small",
    width_mult: float = 1.0,
    num_queries: int | None = None,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown Mask DINO variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]

    return MaskDINO(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        stem_channels=scale_channels(int(spec["stem"]), float(width_mult), min_ch=16, divisor=8),
        p2_channels=scale_channels(int(spec["p2"]), float(width_mult), min_ch=16, divisor=8),
        p3_channels=scale_channels(int(spec["p3"]), float(width_mult), min_ch=16, divisor=8),
        p4_channels=scale_channels(int(spec["p4"]), float(width_mult), min_ch=16, divisor=8),
        hidden_channels=scale_channels(int(spec["hidden"]), float(width_mult), min_ch=16, divisor=8),
        backbone_depth=int(spec["depth"]),
        num_queries=int(spec["queries"]) if num_queries is None else int(num_queries),
        mask_size=int(spec["mask"]),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_mask_dino_instance_segmenter(in_channels=3, num_classes=3, variant="mask_dino_tiny", width_mult=0.5)
    out = m(x)
    print("mask_dino_tiny", {k: tuple(v.shape) for k, v in out.items()})
    loss = sum(v.mean() for v in out.values())
    loss.backward()
    print("ok")
