import torch
from torch import nn

from dlhub.vision.backbones._blocks import ConvBNAct, scale_channels
from dlhub.vision.instance_segmentation._common import (
    BackbonePyramid,
    ContourDecoder,
    InstanceTokenHead,
    check_nchw,
)


class DeepSnake(nn.Module):
    """DeepSnake-style contour refinement instance segmenter."""

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
        num_vertices: int,
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
        self.center_head = nn.Sequential(
            ConvBNAct(int(p2_channels), int(hidden_channels), kernel_size=3, stride=1, act="relu"),
            nn.Conv2d(int(hidden_channels), 1, kernel_size=1),
        )
        self.tokens = InstanceTokenHead(
            int(p4_channels), int(hidden_channels), int(num_instances), depth=2
        )
        self.cls_head = nn.Linear(int(hidden_channels), int(num_classes))
        self.contour = ContourDecoder(
            int(hidden_channels), num_vertices=int(num_vertices), mask_size=int(mask_size)
        )
        self.offset_head = nn.Linear(int(hidden_channels), int(num_vertices) * 2)
        self.num_vertices = int(num_vertices)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = check_nchw(x)
        p2, _, p4 = self.backbone(x)
        center_heatmap = self.center_head(p2)
        tokens = self.tokens(p4)
        b, k, _ = tokens.shape

        cls_logits = self.cls_head(tokens)
        polygons, mask_logits = self.contour(tokens)
        contour_offsets = self.offset_head(tokens).view(b, k, self.num_vertices, 2)
        return {
            "center_heatmap": center_heatmap,
            "cls_logits": cls_logits,
            "polygons": polygons,
            "contour_offsets": contour_offsets,
            "mask_logits": mask_logits,
        }


_VARIANTS: dict[str, dict[str, int]] = {
    "deepsnake_tiny": {
        "stem": 24,
        "p2": 40,
        "p3": 64,
        "p4": 96,
        "hidden": 96,
        "depth": 1,
        "instances": 16,
        "vertices": 16,
        "mask": 16,
    },
    "deepsnake_small": {
        "stem": 24,
        "p2": 48,
        "p3": 80,
        "p4": 128,
        "hidden": 128,
        "depth": 2,
        "instances": 24,
        "vertices": 24,
        "mask": 16,
    },
    "deepsnake_base": {
        "stem": 32,
        "p2": 64,
        "p3": 96,
        "p4": 160,
        "hidden": 160,
        "depth": 3,
        "instances": 32,
        "vertices": 32,
        "mask": 28,
    },
}


def build_deepsnake_instance_segmenter(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "deepsnake_small",
    width_mult: float = 1.0,
    num_instances: int | None = None,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown DeepSnake variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]

    return DeepSnake(
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
        num_vertices=int(spec["vertices"]),
        mask_size=int(spec["mask"]),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_deepsnake_instance_segmenter(
        in_channels=3, num_classes=3, variant="deepsnake_tiny", width_mult=0.5
    )
    out = m(x)
    print("deepsnake_tiny", {k: tuple(v.shape) for k, v in out.items()})
    loss = sum(v.mean() for v in out.values())
    loss.backward()
    print("ok")
