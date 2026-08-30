from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn
from torch.nn import functional as F


def check_bnc(x: torch.Tensor, *, name: str) -> torch.Tensor:
    points = x.to(torch.float32)
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError(f"{name} must have shape (B, N, 3), got {tuple(points.shape)}")
    if points.shape[1] < 2:
        raise ValueError(f"{name} must contain at least two points")
    return points


def _pair(source: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    source = check_bnc(source, name="source")
    target = check_bnc(target, name="target")
    if source.shape[0] != target.shape[0]:
        raise ValueError("source and target batch sizes must match")
    return source, target


def _scaled_width(width: int, width_mult: float) -> int:
    scaled = max(16, int(int(width) * float(width_mult)))
    return ((scaled + 3) // 4) * 4


class _PointEncoder(nn.Module):
    def __init__(self, width: int, depth: int, *, in_features: int = 3):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(in_features, width), nn.GELU()]
        for _ in range(max(0, depth - 1)):
            layers.extend((nn.Linear(width, width), nn.GELU()))
        self.net = nn.Sequential(*layers)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        return self.net(points)


class _PoseHead(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.correction = nn.Sequential(
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, 6),
        )

    def forward(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        source_features: torch.Tensor,
        target_features: torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if weights is None:
            source_descriptor = source_features.mean(dim=1)
            target_descriptor = target_features.mean(dim=1)
        else:
            normalized = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
            source_descriptor = (source_features * normalized[:, :, None]).sum(dim=1)
            target_descriptor = (target_features * normalized[:, :, None]).sum(dim=1)
        raw = self.correction(target_descriptor - source_descriptor)
        centroid_delta = target.mean(dim=1) - source.mean(dim=1)
        translation = centroid_delta + 0.05 * torch.tanh(raw[:, :3])
        rotation = 0.5 * torch.tanh(raw[:, 3:])
        return {"pose6d": torch.cat((translation, rotation), dim=-1)}


class CompactPointNetLK(nn.Module):
    mechanism = "iterative-global-feature-alignment"

    def __init__(self, *, width: int, depth: int):
        super().__init__()
        self.iterations = depth
        self.encoder = _PointEncoder(width, depth)
        self.refinement = nn.ModuleList(
            [nn.Linear(width, width) for _ in range(depth)]
        )
        self.pose_head = _PoseHead(width)

    def _refine(self, features: torch.Tensor) -> torch.Tensor:
        for layer in self.refinement:
            context = features.max(dim=1, keepdim=True).values
            features = features + F.gelu(layer(context)).expand_as(features)
        return features

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        source, target = _pair(source, target)
        source_features = self._refine(self.encoder(source))
        target_features = self._refine(self.encoder(target))
        return self.pose_head(source, target, source_features, target_features)


class CompactDCP(nn.Module):
    mechanism = "cross-attention-correspondence"

    def __init__(self, *, width: int, depth: int):
        super().__init__()
        self.encoder = _PointEncoder(width, depth)
        self.query = nn.Linear(width, width, bias=False)
        self.key = nn.Linear(width, width, bias=False)
        self.pose_head = _PoseHead(width)
        self.last_correspondence: torch.Tensor | None = None

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        source, target = _pair(source, target)
        source_features = self.encoder(source)
        target_features = self.encoder(target)
        scale = source_features.shape[-1] ** -0.5
        logits = self.query(source_features) @ self.key(target_features).transpose(1, 2)
        correspondence = torch.softmax(logits * scale, dim=-1)
        self.last_correspondence = correspondence.detach()
        matched_target = correspondence @ target_features
        return self.pose_head(source, target, source_features, matched_target)


class CompactRegTR(nn.Module):
    mechanism = "joint-source-target-transformer"

    def __init__(self, *, width: int, depth: int):
        super().__init__()
        self.embedding = nn.Linear(3, width)
        self.cloud_embedding = nn.Parameter(torch.zeros(2, width))
        layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=4,
            dim_feedforward=width * 2,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=depth,
            enable_nested_tensor=False,
        )
        self.pose_head = _PoseHead(width)

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        source, target = _pair(source, target)
        source_features = self.embedding(source) + self.cloud_embedding[0]
        target_features = self.embedding(target) + self.cloud_embedding[1]
        source_count = source_features.shape[1]
        encoded = self.transformer(torch.cat((source_features, target_features), dim=1))
        return self.pose_head(
            source,
            target,
            encoded[:, :source_count],
            encoded[:, source_count:],
        )


def _sinkhorn(logits: torch.Tensor, iterations: int) -> torch.Tensor:
    log_transport = logits
    for _ in range(iterations):
        log_transport = log_transport - torch.logsumexp(log_transport, dim=-1, keepdim=True)
        log_transport = log_transport - torch.logsumexp(log_transport, dim=-2, keepdim=True)
    return log_transport.exp()


class CompactRPMNet(nn.Module):
    mechanism = "annealed-sinkhorn-matching"

    def __init__(self, *, width: int, depth: int):
        super().__init__()
        self.sinkhorn_iterations = depth + 2
        self.temperature = nn.Parameter(torch.tensor(1.0))
        self.encoder = _PointEncoder(width, depth)
        self.pose_head = _PoseHead(width)
        self.last_transport: torch.Tensor | None = None

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        source, target = _pair(source, target)
        source_features = F.normalize(self.encoder(source), dim=-1)
        target_features = F.normalize(self.encoder(target), dim=-1)
        temperature = self.temperature.abs().clamp_min(0.05)
        logits = (source_features @ target_features.transpose(1, 2)) / temperature
        transport = _sinkhorn(logits, self.sinkhorn_iterations)
        transport = transport / transport.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        self.last_transport = transport.detach()
        matched_target = transport @ target_features
        return self.pose_head(source, target, source_features, matched_target)


class CompactDeepGMR(nn.Module):
    mechanism = "soft-gaussian-mixture-alignment"

    def __init__(self, *, width: int, depth: int):
        super().__init__()
        self.components = max(4, depth * 2)
        self.encoder = _PointEncoder(width, depth)
        self.assignment = nn.Linear(width, self.components)
        self.pose_head = _PoseHead(width)
        self.last_assignments: tuple[torch.Tensor, torch.Tensor] | None = None

    def _mixture(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        assignments = torch.softmax(self.assignment(features), dim=-1)
        normalizer = assignments.sum(dim=1).clamp_min(1e-6)
        descriptors = torch.einsum("bnk,bnw->bkw", assignments, features)
        return descriptors / normalizer[:, :, None], assignments

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        source, target = _pair(source, target)
        source_components, source_assignments = self._mixture(self.encoder(source))
        target_components, target_assignments = self._mixture(self.encoder(target))
        self.last_assignments = (
            source_assignments.detach(),
            target_assignments.detach(),
        )
        return self.pose_head(
            source,
            target,
            source_components,
            target_components,
        )


class CompactSpinReg(nn.Module):
    mechanism = "cylindrical-spin-descriptor"

    def __init__(self, *, width: int, depth: int):
        super().__init__()
        self.encoder = _PointEncoder(width, depth, in_features=5)
        self.pose_head = _PoseHead(width)

    @staticmethod
    def _spin_features(points: torch.Tensor) -> torch.Tensor:
        centered = points - points.mean(dim=1, keepdim=True)
        xy_radius = centered[..., :2].square().sum(dim=-1).sqrt()
        radius = centered.square().sum(dim=-1).sqrt()
        angle = torch.atan2(centered[..., 1], centered[..., 0])
        return torch.stack(
            (radius, xy_radius, centered[..., 2], torch.sin(angle), torch.cos(angle)),
            dim=-1,
        )

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        source, target = _pair(source, target)
        source_features = self.encoder(self._spin_features(source))
        target_features = self.encoder(self._spin_features(target))
        return self.pose_head(source, target, source_features, target_features)


class CompactCoFiNet(nn.Module):
    mechanism = "coarse-to-fine-correspondence"

    def __init__(self, *, width: int, depth: int):
        super().__init__()
        self.encoder = _PointEncoder(width, depth)
        self.fine_fusion = nn.Linear(width * 2, width)
        self.pose_head = _PoseHead(width)
        self.last_coarse_correspondence: torch.Tensor | None = None

    @staticmethod
    def _coarse_points(points: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        centered = points - points.mean(dim=1, keepdim=True)
        order = centered.square().sum(dim=-1).argsort(dim=1)
        ordered = features.gather(
            1,
            order[:, :, None].expand(-1, -1, features.shape[-1]),
        )
        stride = max(1, points.shape[1] // 16)
        return ordered[:, ::stride]

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        source, target = _pair(source, target)
        source_features = self.encoder(source)
        target_features = self.encoder(target)
        source_coarse = self._coarse_points(source, source_features)
        target_coarse = self._coarse_points(target, target_features)
        logits = source_coarse @ target_coarse.transpose(1, 2)
        correspondence = torch.softmax(logits * source_features.shape[-1] ** -0.5, dim=-1)
        self.last_coarse_correspondence = correspondence.detach()
        coarse_context = (correspondence @ target_coarse).mean(dim=1, keepdim=True)
        coarse_context = coarse_context.expand(-1, source_features.shape[1], -1)
        refined_source = F.gelu(
            self.fine_fusion(torch.cat((source_features, coarse_context), dim=-1))
        )
        return self.pose_head(source, target, refined_source, target_features)


class CompactGeoFormer(nn.Module):
    mechanism = "geometry-biased-attention"

    def __init__(self, *, width: int, depth: int):
        super().__init__()
        self.encoder = _PointEncoder(width, depth)
        self.geometry_fusion = nn.Linear(width * 2, width)
        self.cross_attention = nn.MultiheadAttention(
            width,
            num_heads=4,
            dropout=0.0,
            batch_first=True,
        )
        self.pose_head = _PoseHead(width)

    def _geometry(self, points: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        distances = torch.cdist(points, points)
        scale = distances.detach().mean(dim=(1, 2), keepdim=True).clamp_min(1e-4)
        neighbors = torch.softmax(-distances / scale, dim=-1) @ features
        return F.gelu(self.geometry_fusion(torch.cat((features, neighbors), dim=-1)))

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        source, target = _pair(source, target)
        source_features = self._geometry(source, self.encoder(source))
        target_features = self._geometry(target, self.encoder(target))
        matched_target, _ = self.cross_attention(
            source_features,
            target_features,
            target_features,
            need_weights=False,
        )
        return self.pose_head(source, target, source_features, matched_target)


class CompactPredatorReg(nn.Module):
    mechanism = "overlap-weighted-correspondence"

    def __init__(self, *, width: int, depth: int):
        super().__init__()
        self.encoder = _PointEncoder(width, depth)
        self.overlap_head = nn.Linear(width * 2, 1)
        self.pose_head = _PoseHead(width)
        self.last_overlap: torch.Tensor | None = None

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        source, target = _pair(source, target)
        source_features = self.encoder(source)
        target_features = self.encoder(target)
        similarities = source_features @ target_features.transpose(1, 2)
        correspondence = torch.softmax(similarities * source_features.shape[-1] ** -0.5, dim=-1)
        matched_target = correspondence @ target_features
        overlap = torch.sigmoid(
            self.overlap_head(torch.cat((source_features, matched_target), dim=-1)).squeeze(-1)
        )
        self.last_overlap = overlap.detach()
        return self.pose_head(
            source,
            target,
            source_features,
            matched_target,
            overlap,
        )


class _SelectiveScan(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.projection = nn.Linear(width, width * 2)
        self.norm = nn.LayerNorm(width)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        gates, candidates = self.projection(tokens).chunk(2, dim=-1)
        gates = torch.sigmoid(gates)
        candidates = torch.tanh(candidates)
        state = torch.zeros_like(tokens[:, 0])
        outputs = []
        for gate, candidate in zip(
            gates.unbind(dim=1), candidates.unbind(dim=1), strict=True
        ):
            state = gate * state + (1.0 - gate) * candidate
            outputs.append(state)
        return self.norm(tokens + torch.stack(outputs, dim=1))


class CompactMambaReg(nn.Module):
    mechanism = "radial-order-selective-scan"

    def __init__(self, *, width: int, depth: int):
        super().__init__()
        self.encoder = nn.Linear(3, width)
        self.scans = nn.ModuleList([_SelectiveScan(width) for _ in range(depth)])
        self.pose_head = _PoseHead(width)

    def _encode(self, points: torch.Tensor) -> torch.Tensor:
        centered = points - points.mean(dim=1, keepdim=True)
        order = centered.square().sum(dim=-1).argsort(dim=1)
        ordered = points.gather(1, order[:, :, None].expand(-1, -1, 3))
        features = self.encoder(ordered)
        for scan in self.scans:
            features = scan(features)
        return features

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        source, target = _pair(source, target)
        return self.pose_head(
            source,
            target,
            self._encode(source),
            self._encode(target),
        )


RegistrationBuilder = Callable[..., nn.Module]

_FAMILY_BUILDERS: dict[str, RegistrationBuilder] = {
    "pointnetlk": CompactPointNetLK,
    "dcp": CompactDCP,
    "regtr": CompactRegTR,
    "rpmnet": CompactRPMNet,
    "deepgmr": CompactDeepGMR,
    "spinreg": CompactSpinReg,
    "cofinet_reg": CompactCoFiNet,
    "geoformer_reg": CompactGeoFormer,
    "predator_reg": CompactPredatorReg,
    "mambareg": CompactMambaReg,
}


def build_compact_registration_model(
    *,
    family: str,
    variants: dict[str, dict[str, int]],
    variant: str,
    width_mult: float = 1.0,
    **_: object,
) -> nn.Module:
    if variant not in variants:
        raise KeyError(f"Unknown {family} registration variant: {variant!r}")
    try:
        builder = _FAMILY_BUILDERS[family]
    except KeyError as error:
        raise KeyError(f"Unknown registration mechanism family: {family!r}") from error
    spec = variants[variant]
    return builder(
        width=_scaled_width(spec["width"], width_mult),
        depth=int(spec["depth"]),
    )


def validate_registration_model(builder: Callable[..., nn.Module], variant: str) -> None:
    model = builder(variant=variant, width_mult=0.5)
    output = model(torch.randn(2, 128, 3), torch.randn(2, 128, 3))
    print(variant, model.mechanism, tuple(output["pose6d"].shape))


__all__ = [
    "build_compact_registration_model",
    "check_bnc",
    "validate_registration_model",
]
