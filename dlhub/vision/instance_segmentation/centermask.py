import torch
from torch import nn

from dlhub.vision.backbones._blocks import ConvBNAct, scale_channels
from dlhub.vision.instance_segmentation._common import BackboneLowDet, ProtoNet, check_nchw


class CenterMask(nn.Module):
    """CenterMask-style instance segmentation (toy-first).

    Center-based detection heads + prototype-mask representation.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        num_protos: int = 32,
        stem_channels: int = 24,
        low_channels: int = 48,
        det_channels: int = 96,
        backbone_depth: int = 2,
        head_channels: int = 96,
        head_convs: int = 2,
        proto_depth: int = 3,
    ) -> None:
        super().__init__()
        nc = int(num_classes)
        if nc <= 0:
            raise ValueError("num_classes must be > 0")
        np = int(num_protos)
        if np <= 0:
            raise ValueError("num_protos must be > 0")

        self.backbone = BackboneLowDet(
            in_channels=int(in_channels),
            stem_channels=int(stem_channels),
            low_channels=int(low_channels),
            det_channels=int(det_channels),
            depth=int(backbone_depth),
            act="relu",
        )
        self.proto = ProtoNet(int(low_channels), np, depth=int(proto_depth))

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

        self.heatmap = nn.Conv2d(int(head_channels), nc, kernel_size=3, padding=1)
        self.wh = nn.Conv2d(int(head_channels), 2, kernel_size=3, padding=1)
        self.offset = nn.Conv2d(int(head_channels), 2, kernel_size=3, padding=1)

        self.coeff = nn.Conv2d(int(head_channels), np, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = check_nchw(x)
        low, det = self.backbone(x)
        proto = self.proto(low)  # (B,P,H/4,W/4)

        t = self.tower(det)
        heatmap = self.heatmap(t)
        wh = self.wh(t)
        offset = self.offset(t)
        coeffs = self.coeff(t)  # (B,P,H/8,W/8)

        b, p, h4, w4 = proto.shape
        slots = coeffs.shape[-2] * coeffs.shape[-1]
        coeff_flat = coeffs.permute(0, 2, 3, 1).reshape(b, slots, p)
        proto_flat = proto.reshape(b, p, h4 * w4)
        mask_logits = torch.bmm(coeff_flat, proto_flat).view(b, slots, h4, w4)
        return {
            "heatmap": heatmap,
            "wh": wh,
            "offset": offset,
            "mask_logits": mask_logits,
            "proto": proto,
            "mask_coeffs": coeffs,
        }


_VARIANTS: dict[str, dict] = {
    "centermask_tiny": {"stem": 24, "low": 40, "det": 80, "depth": 1, "head": 80, "protos": 16},
    "centermask_small": {"stem": 24, "low": 48, "det": 96, "depth": 2, "head": 96, "protos": 32},
    "centermask_base": {"stem": 32, "low": 64, "det": 128, "depth": 3, "head": 128, "protos": 48},
}


def build_centermask_instance_segmenter(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "centermask_small",
    width_mult: float = 1.0,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown CenterMask variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    stem = scale_channels(int(spec["stem"]), float(width_mult), min_ch=16, divisor=8)
    low = scale_channels(int(spec["low"]), float(width_mult), min_ch=16, divisor=8)
    det = scale_channels(int(spec["det"]), float(width_mult), min_ch=16, divisor=8)
    head = scale_channels(int(spec["head"]), float(width_mult), min_ch=16, divisor=8)
    protos = max(8, int(round(int(spec["protos"]) * float(width_mult))))
    return CenterMask(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        num_protos=int(protos),
        stem_channels=int(stem),
        low_channels=int(low),
        det_channels=int(det),
        backbone_depth=int(spec["depth"]),
        head_channels=int(head),
        head_convs=2,
        proto_depth=3,
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_centermask_instance_segmenter(
        in_channels=3, num_classes=3, variant="centermask_tiny", width_mult=0.5
    )
    out = m(x)
    print("centermask_tiny", {k: tuple(v.shape) for k, v in out.items()})
    loss = sum(v.mean() for v in out.values())
    loss.backward()
    print("ok")
