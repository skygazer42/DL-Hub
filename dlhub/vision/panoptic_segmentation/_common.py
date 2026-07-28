import torch
import torch.nn.functional as F
from torch import nn

from dlhub.vision.backbones._blocks import ConvBNAct


def check_nchw(x: torch.Tensor) -> torch.Tensor:
    x = x.to(torch.float32)
    if x.ndim != 4:
        raise ValueError(f"Expected input shape (B, C, H, W), got {tuple(x.shape)}")
    return x


class BackboneC2C3C4C5(nn.Module):
    """Tiny conv backbone that returns (C2, C3, C4, C5).

    Strides: approximately (/4, /8, /16, /32) relative to input.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        stem_channels: int,
        c2_channels: int,
        c3_channels: int,
        c4_channels: int,
        c5_channels: int,
        depth: int,
        act: str = "relu",
    ) -> None:
        super().__init__()
        c_in = int(in_channels)
        stem = int(stem_channels)
        c2 = int(c2_channels)
        c3 = int(c3_channels)
        c4 = int(c4_channels)
        c5 = int(c5_channels)
        d = int(depth)
        if d <= 0:
            raise ValueError("depth must be > 0")

        self.stem = nn.Sequential(
            ConvBNAct(c_in, stem, kernel_size=3, stride=2, act=act),  # /2
            ConvBNAct(stem, c2, kernel_size=3, stride=2, act=act),  # /4
        )

        def stage(in_ch: int, out_ch: int) -> nn.Sequential:
            layers: list[nn.Module] = [ConvBNAct(in_ch, out_ch, kernel_size=3, stride=2, act=act)]
            for _ in range(d - 1):
                layers.append(ConvBNAct(out_ch, out_ch, kernel_size=3, stride=1, act=act))
            return nn.Sequential(*layers)

        self.stage3 = stage(c2, c3)
        self.stage4 = stage(c3, c4)
        self.stage5 = stage(c4, c5)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        c2 = self.stem(x)
        c3 = self.stage3(c2)
        c4 = self.stage4(c3)
        c5 = self.stage5(c4)
        return c2, c3, c4, c5


class FPN4(nn.Module):
    """Minimal 4-level FPN: (C2..C5) -> (P2..P5)."""

    def __init__(
        self, in_channels: tuple[int, int, int, int], out_channels: int, *, act: str = "relu"
    ) -> None:
        super().__init__()
        c2, c3, c4, c5 = (int(x) for x in in_channels)
        out = int(out_channels)

        self.l2 = nn.Conv2d(c2, out, kernel_size=1)
        self.l3 = nn.Conv2d(c3, out, kernel_size=1)
        self.l4 = nn.Conv2d(c4, out, kernel_size=1)
        self.l5 = nn.Conv2d(c5, out, kernel_size=1)

        self.p2 = ConvBNAct(out, out, kernel_size=3, stride=1, act=act)
        self.p3 = ConvBNAct(out, out, kernel_size=3, stride=1, act=act)
        self.p4 = ConvBNAct(out, out, kernel_size=3, stride=1, act=act)
        self.p5 = ConvBNAct(out, out, kernel_size=3, stride=1, act=act)

    def forward(
        self, c2: torch.Tensor, c3: torch.Tensor, c4: torch.Tensor, c5: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        p5 = self.l5(c5)
        p4 = self.l4(c4) + F.interpolate(p5, size=c4.shape[-2:], mode="nearest")
        p3 = self.l3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="nearest")
        p2 = self.l2(c2) + F.interpolate(p3, size=c2.shape[-2:], mode="nearest")
        return self.p2(p2), self.p3(p3), self.p4(p4), self.p5(p5)


class ProtoNet(nn.Module):
    """Prototype mask generator used by many panoptic models (compact-first)."""

    def __init__(self, in_ch: int, proto_ch: int, *, depth: int = 3, act: str = "relu") -> None:
        super().__init__()
        c_in = int(in_ch)
        p = int(proto_ch)
        d = int(depth)
        if p <= 0:
            raise ValueError("proto_ch must be > 0")
        if d <= 0:
            raise ValueError("depth must be > 0")

        layers: list[nn.Module] = []
        ch = c_in
        for _ in range(d - 1):
            layers.append(ConvBNAct(ch, ch, kernel_size=3, stride=1, act=act))
        layers.append(nn.Conv2d(ch, p, kernel_size=1, bias=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BackboneLowDet(nn.Module):
    """Backbone that returns (low_stride4, det_stride8) feature maps."""

    def __init__(
        self,
        *,
        in_channels: int,
        stem_channels: int,
        low_channels: int,
        det_channels: int,
        depth: int,
        act: str = "relu",
    ) -> None:
        super().__init__()
        c_in = int(in_channels)
        stem = int(stem_channels)
        low = int(low_channels)
        det = int(det_channels)
        d = int(depth)
        if d <= 0:
            raise ValueError("depth must be > 0")

        self.stem = ConvBNAct(c_in, stem, kernel_size=3, stride=2, act=act)  # /2
        self.to_low = ConvBNAct(stem, low, kernel_size=3, stride=2, act=act)  # /4

        layers: list[nn.Module] = [ConvBNAct(low, det, kernel_size=3, stride=2, act=act)]  # /8
        for _ in range(d - 1):
            layers.append(ConvBNAct(det, det, kernel_size=3, stride=1, act=act))
        self.det = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        low = self.to_low(x)
        det = self.det(low)
        return low, det


class ConvTower(nn.Module):
    """A small stack of Conv-BN-Act blocks."""

    def __init__(self, channels: int, *, num_convs: int = 2, act: str = "relu") -> None:
        super().__init__()
        c = int(channels)
        n = int(num_convs)
        if c <= 0:
            raise ValueError("channels must be > 0")
        if n <= 0:
            raise ValueError("num_convs must be > 0")
        self.net = nn.Sequential(
            *[ConvBNAct(c, c, kernel_size=3, stride=1, act=act) for _ in range(n)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DensePredHead(nn.Module):
    """Dense prediction head for one-stage panoptic/instance models (compact-first)."""

    def __init__(
        self,
        *,
        in_ch: int,
        num_classes: int,
        num_anchors: int,
        num_protos: int,
        head_channels: int,
        num_convs: int = 2,
        act: str = "relu",
    ) -> None:
        super().__init__()
        c_in = int(in_ch)
        nc = int(num_classes)
        na = int(num_anchors)
        np = int(num_protos)
        hc = int(head_channels)
        n = int(num_convs)
        if nc <= 0:
            raise ValueError("num_classes must be > 0")
        if na <= 0:
            raise ValueError("num_anchors must be > 0")
        if np <= 0:
            raise ValueError("num_protos must be > 0")
        if n <= 0:
            raise ValueError("num_convs must be > 0")

        layers: list[nn.Module] = [ConvBNAct(c_in, hc, kernel_size=3, stride=1, act=act)]
        for _ in range(n - 1):
            layers.append(ConvBNAct(hc, hc, kernel_size=3, stride=1, act=act))
        self.tower = nn.Sequential(*layers)

        self.cls_logits = nn.Conv2d(hc, na * nc, kernel_size=3, padding=1)
        self.box_pred = nn.Conv2d(hc, na * 4, kernel_size=3, padding=1)
        self.mask_coeff = nn.Conv2d(hc, na * np, kernel_size=3, padding=1)

        nn.init.normal_(self.cls_logits.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.cls_logits.bias, 0.0)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        t = self.tower(x)
        return self.cls_logits(t), self.box_pred(t), self.mask_coeff(t)


def upsample_like(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    if x.shape[-2:] == ref.shape[-2:]:
        return x
    return F.interpolate(x, size=ref.shape[-2:], mode="nearest")


def masks_from_prototypes(proto: torch.Tensor, coeff: torch.Tensor) -> torch.Tensor:
    """Combine prototype masks and coefficients into mask logits.

    proto: (B, P, H, W)
    coeff: (B, N, P)
    returns: (B, N, H, W)
    """
    if proto.ndim != 4:
        raise ValueError(f"proto must be (B,P,H,W), got {tuple(proto.shape)}")
    if coeff.ndim != 3:
        raise ValueError(f"coeff must be (B,N,P), got {tuple(coeff.shape)}")
    b, p, h, w = proto.shape
    if coeff.shape[0] != b or coeff.shape[2] != p:
        raise ValueError("coeff batch/proto mismatch")
    proto_flat = proto.reshape(b, p, h * w)
    mask_flat = torch.bmm(coeff, proto_flat)
    return mask_flat.view(b, coeff.shape[1], h, w)


def fuse_panoptic(
    semantic_logits: torch.Tensor,
    instance_masks: torch.Tensor,
    instance_scores: torch.Tensor,
    *,
    thing_offset: int,
) -> torch.Tensor:
    """Compact panoptic fusion: argmax semantic + add instances by score ordering.

    Returns a panoptic id map (B,H,W) with:
    - stuff classes: [0, thing_offset)
    - things instances: thing_offset + instance_index
    """
    if semantic_logits.ndim != 4:
        raise ValueError("semantic_logits must be (B,C,H,W)")
    if instance_masks.ndim != 4:
        raise ValueError("instance_masks must be (B,N,H,W)")
    if instance_scores.ndim != 2:
        raise ValueError("instance_scores must be (B,N)")

    b, _, h, w = semantic_logits.shape
    if instance_masks.shape[0] != b or instance_masks.shape[-2:] != (h, w):
        instance_masks = F.interpolate(instance_masks, size=(h, w), mode="nearest")
    sem = semantic_logits.argmax(dim=1).to(torch.int64)  # (B,H,W)
    pan = sem.clone()

    for bi in range(b):
        order = torch.argsort(instance_scores[bi], descending=True)
        for rank, idx in enumerate(order.tolist()):
            mask = instance_masks[bi, idx] > 0.0
            pan[bi][mask] = int(thing_offset) + int(rank)
    return pan


__all__ = [
    "BackboneC2C3C4C5",
    "BackboneLowDet",
    "ConvTower",
    "DensePredHead",
    "FPN4",
    "ProtoNet",
    "check_nchw",
    "fuse_panoptic",
    "masks_from_prototypes",
    "upsample_like",
]
