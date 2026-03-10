import torch
import torch.nn.functional as F
from torch import nn

from dlhub.vision.backbones._blocks import scale_channels
from dlhub.vision.instance_segmentation._common import (
    BackboneLowDet,
    DensePredHead,
    ProtoNet,
    check_nchw,
)


class MEInst(nn.Module):
    """MEInst-style mask encoding instance segmenter."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        stem_channels: int,
        low_channels: int,
        det_channels: int,
        head_channels: int,
        backbone_depth: int,
        num_anchors: int,
        num_codes: int,
    ) -> None:
        super().__init__()
        self.backbone = BackboneLowDet(
            in_channels=int(in_channels),
            stem_channels=int(stem_channels),
            low_channels=int(low_channels),
            det_channels=int(det_channels),
            depth=int(backbone_depth),
        )
        self.proto_net = ProtoNet(int(low_channels), int(num_codes), depth=3)
        self.pred_head = DensePredHead(
            in_ch=int(det_channels),
            num_classes=int(num_classes),
            num_anchors=int(num_anchors),
            num_protos=int(num_codes),
            head_channels=int(head_channels),
            num_convs=2,
        )
        self.num_anchors = int(num_anchors)
        self.num_codes = int(num_codes)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = check_nchw(x)
        low, det = self.backbone(x)
        codebook = self.proto_net(low)
        cls_logits, bbox_deltas, mask_code_map = self.pred_head(det)

        b = x.shape[0]
        mask_latents = F.adaptive_avg_pool2d(mask_code_map, (1, 1)).view(
            b, self.num_anchors, self.num_codes
        )
        mask_logits = torch.einsum("bkp,bphw->bkhw", mask_latents, codebook)
        code_mean = codebook.mean(dim=(-2, -1))
        return {
            "cls_logits": cls_logits,
            "bbox_deltas": bbox_deltas,
            "mask_latents": mask_latents,
            "codebook": codebook,
            "code_mean": code_mean,
            "mask_logits": mask_logits,
        }


_VARIANTS: dict[str, dict[str, int]] = {
    "meinst_tiny": {
        "stem": 24,
        "low": 40,
        "det": 72,
        "head": 72,
        "depth": 1,
        "anchors": 8,
        "codes": 16,
    },
    "meinst_small": {
        "stem": 24,
        "low": 48,
        "det": 96,
        "head": 96,
        "depth": 2,
        "anchors": 12,
        "codes": 24,
    },
    "meinst_base": {
        "stem": 32,
        "low": 64,
        "det": 128,
        "head": 128,
        "depth": 3,
        "anchors": 16,
        "codes": 32,
    },
}


def build_meinst_instance_segmenter(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "meinst_small",
    width_mult: float = 1.0,
    num_anchors: int | None = None,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown MEInst variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]

    return MEInst(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        stem_channels=scale_channels(int(spec["stem"]), float(width_mult), min_ch=16, divisor=8),
        low_channels=scale_channels(int(spec["low"]), float(width_mult), min_ch=16, divisor=8),
        det_channels=scale_channels(int(spec["det"]), float(width_mult), min_ch=16, divisor=8),
        head_channels=scale_channels(int(spec["head"]), float(width_mult), min_ch=16, divisor=8),
        backbone_depth=int(spec["depth"]),
        num_anchors=int(spec["anchors"]) if num_anchors is None else int(num_anchors),
        num_codes=int(spec["codes"]),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_meinst_instance_segmenter(
        in_channels=3, num_classes=3, variant="meinst_tiny", width_mult=0.5
    )
    out = m(x)
    print("meinst_tiny", {k: tuple(v.shape) for k, v in out.items()})
    loss = sum(v.mean() for v in out.values())
    loss.backward()
    print("ok")
