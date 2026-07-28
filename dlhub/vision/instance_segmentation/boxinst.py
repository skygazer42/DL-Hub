import torch
from torch import nn

from dlhub.vision.backbones._blocks import ConvBNAct, scale_channels
from dlhub.vision.instance_segmentation._common import BackboneLowDet, check_nchw


class BoxInst(nn.Module):
    """BoxInst-style instance segmentation (compact-first).

    BoxInst trains masks using only boxes; here we focus on the model skeleton:
    dense box prediction + dynamic masks from a mask feature map.
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

        self.cls = nn.Conv2d(int(head_channels), nc, kernel_size=3, padding=1)
        self.box = nn.Conv2d(int(head_channels), 4, kernel_size=3, padding=1)
        self.kernel = nn.Conv2d(int(head_channels), mc, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = check_nchw(x)
        low, det = self.backbone(x)
        mf = self.mask_feat(low)

        t = self.tower(det)
        cls_logits = self.cls(t)
        bbox = self.box(t)
        kernels = self.kernel(t)

        b, m, h4, w4 = mf.shape
        slots = kernels.shape[-2] * kernels.shape[-1]
        ker_flat = kernels.permute(0, 2, 3, 1).reshape(b, slots, m)
        mf_flat = mf.reshape(b, m, h4 * w4)
        mask_logits = torch.bmm(ker_flat, mf_flat).view(b, slots, h4, w4)
        return {
            "cls_logits": cls_logits,
            "bbox_deltas": bbox,
            "mask_logits": mask_logits,
            "mask_feat": mf,
            "mask_kernels": kernels,
        }


_VARIANTS: dict[str, dict] = {
    "boxinst_tiny": {"stem": 24, "low": 40, "det": 80, "depth": 1, "head": 80, "mask": 24},
    "boxinst_small": {"stem": 24, "low": 48, "det": 96, "depth": 2, "head": 96, "mask": 32},
    "boxinst_base": {"stem": 32, "low": 64, "det": 128, "depth": 3, "head": 128, "mask": 48},
}


def build_boxinst_instance_segmenter(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "boxinst_small",
    width_mult: float = 1.0,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown BoxInst variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    stem = scale_channels(int(spec["stem"]), float(width_mult), min_ch=16, divisor=8)
    low = scale_channels(int(spec["low"]), float(width_mult), min_ch=16, divisor=8)
    det = scale_channels(int(spec["det"]), float(width_mult), min_ch=16, divisor=8)
    head = scale_channels(int(spec["head"]), float(width_mult), min_ch=16, divisor=8)
    mask = scale_channels(int(spec["mask"]), float(width_mult), min_ch=16, divisor=8)
    return BoxInst(
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
    m = build_boxinst_instance_segmenter(
        in_channels=3, num_classes=3, variant="boxinst_tiny", width_mult=0.5
    )
    out = m(x)
    print("boxinst_tiny", {k: tuple(v.shape) for k, v in out.items()})
    loss = sum(v.mean() for v in out.values())
    loss.backward()
    print("ok")
