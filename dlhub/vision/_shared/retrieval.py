from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


def check_nchw(x: torch.Tensor) -> torch.Tensor:
    image = x.to(torch.float32)
    if image.ndim != 4:
        raise ValueError(f"Expected input shape (B, C, H, W), got {tuple(image.shape)}")
    return image


class TinyRetrievalEncoder(nn.Module):
    def __init__(self, *, in_channels: int, width: int, depth: int) -> None:
        super().__init__()
        channels = int(width)
        layers: list[nn.Module] = [
            nn.Conv2d(int(in_channels), channels, 3, 2, 1),
            nn.GELU(),
        ]
        for _ in range(max(1, int(depth))):
            layers.extend(
                (
                    nn.Conv2d(channels, channels * 2, 3, 2, 1),
                    nn.GELU(),
                )
            )
            channels *= 2
        self.net = nn.Sequential(*layers)
        self.out_channels = channels

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(check_nchw(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(x)
        return F.normalize(F.adaptive_avg_pool2d(features, 1).flatten(1), dim=1)


class _AveragePool(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.out_dim = channels

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features.mean(dim=(-2, -1))


class _GeMPool(nn.Module):
    def __init__(self, channels: int, *, initial_p: float):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(float(initial_p)))
        self.out_dim = channels

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        p = self.p.clamp(1.0, 8.0)
        return features.clamp_min(1e-6).pow(p).mean(dim=(-2, -1)).pow(1.0 / p)


class _AttentionPool(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.attention = nn.Conv2d(channels, 1, kernel_size=1)
        self.out_dim = channels
        self.last_attention: torch.Tensor | None = None

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.attention(features).flatten(2), dim=-1)
        self.last_attention = weights.detach()
        tokens = features.flatten(2)
        return (tokens * weights).sum(dim=-1)


class _LocalGlobalPool(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.saliency = nn.Conv2d(channels, 1, kernel_size=1)
        self.out_dim = channels * 2

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        global_descriptor = features.mean(dim=(-2, -1))
        weights = torch.softmax(self.saliency(features).flatten(2), dim=-1)
        local_descriptor = (features.flatten(2) * weights).sum(dim=-1)
        return torch.cat((global_descriptor, local_descriptor), dim=-1)


class _RegionalPool(nn.Module):
    def __init__(self, channels: int, *, grid: int):
        super().__init__()
        self.grid = int(grid)
        self.out_dim = channels * self.grid * self.grid

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return F.adaptive_avg_pool2d(features, self.grid).flatten(1)


class _MultiScalePool(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.out_dim = channels * 5

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        global_descriptor = F.adaptive_avg_pool2d(features, 1).flatten(1)
        regional = F.adaptive_avg_pool2d(features, 2).flatten(1)
        return torch.cat((global_descriptor, regional), dim=-1)


class _NetVLADPool(nn.Module):
    def __init__(self, channels: int, *, clusters: int):
        super().__init__()
        self.clusters = int(clusters)
        self.assignment = nn.Conv2d(channels, self.clusters, kernel_size=1)
        self.centers = nn.Parameter(torch.randn(self.clusters, channels) * 0.02)
        self.out_dim = channels * self.clusters
        self.last_assignment: torch.Tensor | None = None

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        batch, channels, _, _ = features.shape
        tokens = features.flatten(2).transpose(1, 2)
        assignment = torch.softmax(self.assignment(features).flatten(2), dim=1)
        self.last_assignment = assignment.detach()
        residuals = tokens[:, None] - self.centers[None, :, None]
        aggregated = (residuals * assignment[:, :, :, None]).sum(dim=2)
        aggregated = F.normalize(aggregated, dim=-1)
        return F.normalize(aggregated.reshape(batch, self.clusters * channels), dim=-1)


class _TransformerPool(nn.Module):
    def __init__(self, channels: int, *, depth: int):
        super().__init__()
        self.class_token = nn.Parameter(torch.zeros(1, 1, channels))
        self.position_projection = nn.Linear(2, channels)
        layer = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=4,
            dim_feedforward=channels * 2,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=max(1, int(depth)),
            enable_nested_tensor=False,
        )
        self.out_dim = channels

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = features.shape
        rows = torch.linspace(-1.0, 1.0, height, device=features.device, dtype=features.dtype)
        columns = torch.linspace(
            -1.0, 1.0, width, device=features.device, dtype=features.dtype
        )
        row_grid, column_grid = torch.meshgrid(rows, columns, indexing="ij")
        positions = torch.stack((column_grid, row_grid), dim=-1).reshape(1, -1, 2)
        tokens = features.flatten(2).transpose(1, 2)
        tokens = tokens + self.position_projection(positions)
        class_token = self.class_token.expand(batch, -1, -1)
        return self.encoder(torch.cat((class_token, tokens), dim=1))[:, 0]


class _SelectiveScanPool(nn.Module):
    def __init__(self, channels: int, *, depth: int):
        super().__init__()
        self.projections = nn.ModuleList(
            [nn.Linear(channels, channels * 2) for _ in range(max(1, depth))]
        )
        self.norm = nn.LayerNorm(channels)
        self.out_dim = channels

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        tokens = features.flatten(2).transpose(1, 2)
        for projection in self.projections:
            gates, candidates = projection(tokens).chunk(2, dim=-1)
            gates = torch.sigmoid(gates)
            candidates = torch.tanh(candidates)
            state = torch.zeros_like(tokens[:, 0])
            states = []
            for gate, candidate in zip(
                gates.unbind(dim=1), candidates.unbind(dim=1), strict=True
            ):
                state = gate * state + (1.0 - gate) * candidate
                states.append(state)
            tokens = self.norm(tokens + torch.stack(states, dim=1))
        return tokens.mean(dim=1)


class _BilinearPool(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        reduced = min(32, channels)
        self.left = nn.Conv2d(channels, reduced, kernel_size=1)
        self.right = nn.Conv2d(channels, reduced, kernel_size=1)
        self.out_dim = reduced * reduced

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        left = self.left(features).flatten(2)
        right = self.right(features).flatten(2)
        bilinear = left @ right.transpose(1, 2) / left.shape[-1]
        signed_root = bilinear.sign() * bilinear.abs().clamp_min(1e-8).sqrt()
        return F.normalize(signed_root.flatten(1), dim=-1)


class _MixedPool(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.spatial_mixer = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
        )
        self.out_dim = channels * 2

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        mixed = F.gelu(self.spatial_mixer(features))
        return torch.cat((mixed.mean(dim=(-2, -1)), mixed.amax(dim=(-2, -1))), dim=-1)


class _TokenPartPool(nn.Module):
    def __init__(self, channels: int, *, parts: int):
        super().__init__()
        self.part_queries = nn.Parameter(torch.randn(parts, channels) * 0.02)
        self.out_dim = parts * channels
        self.last_part_attention: torch.Tensor | None = None

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        tokens = features.flatten(2).transpose(1, 2)
        logits = torch.einsum("pc,bsc->bps", self.part_queries, tokens)
        attention = torch.softmax(logits * features.shape[1] ** -0.5, dim=-1)
        self.last_part_attention = attention.detach()
        parts = attention @ tokens
        return parts.flatten(1)


@dataclass(frozen=True)
class _RetrievalMechanism:
    pooling: str
    mechanism: str
    scoring: str = "cosine"
    context: bool = False
    option: int = 0


_MECHANISMS = {
    "arc": _RetrievalMechanism("attention", "attention-with-angular-scoring", "angular"),
    "clipret": _RetrievalMechanism("attention", "context-conditioned-attention", context=True),
    "contrastive": _RetrievalMechanism("average", "temperature-scaled-contrastive", "temperature"),
    "delg": _RetrievalMechanism("local_global", "local-global-saliency-aggregation"),
    "gem": _RetrievalMechanism("gem", "learnable-generalized-mean", option=3),
    "netvlad": _RetrievalMechanism("vlad", "soft-residual-vlad", option=4),
    "pairret": _RetrievalMechanism("average", "learned-pairwise-matching", "pairwise"),
    "proxy": _RetrievalMechanism("attention", "learned-proxy-refinement", "proxy"),
    "regional": _RetrievalMechanism("regional", "regional-grid-aggregation", option=2),
    "transformerret": _RetrievalMechanism("transformer", "spatial-token-transformer"),
    "apgem_vpr": _RetrievalMechanism("gem", "place-adaptive-generalized-mean", option=4),
    "cosplace": _RetrievalMechanism("attention", "place-proxy-cosine-refinement", "proxy"),
    "delg_vpr": _RetrievalMechanism("local_global", "place-local-global-saliency"),
    "geoclip_vpr": _RetrievalMechanism(
        "regional", "geo-context-regional-fusion", context=True, option=2
    ),
    "mambavpr": _RetrievalMechanism("scan", "spatial-selective-scan"),
    "mixvpr": _RetrievalMechanism("mixed", "depthwise-spatial-feature-mixing"),
    "pairvpr": _RetrievalMechanism(
        "local_global", "local-global-pairwise-matching", "pairwise"
    ),
    "patchnetvlad": _RetrievalMechanism(
        "vlad", "patch-level-soft-residual-vlad", option=6
    ),
    "regionvpr": _RetrievalMechanism("regional", "place-region-grid-aggregation", option=3),
    "transvpr": _RetrievalMechanism("transformer", "place-token-transformer"),
    "bilinear_fgret": _RetrievalMechanism("bilinear", "bilinear-part-interaction"),
    "descriptor_fgret": _RetrievalMechanism("average", "normalized-global-descriptor"),
    "fgclip_retr": _RetrievalMechanism(
        "token_parts", "context-conditioned-part-tokens", context=True, option=4
    ),
    "granule_retr": _RetrievalMechanism("multiscale", "multi-granularity-pyramid"),
    "mamba_fgret": _RetrievalMechanism("scan", "fine-grained-selective-scan"),
    "partvlad": _RetrievalMechanism("vlad", "part-residual-vlad", option=5),
    "prompt_fgret": _RetrievalMechanism(
        "attention", "prompt-conditioned-attention", context=True
    ),
    "regional_fgret": _RetrievalMechanism(
        "regional", "fine-grained-region-grid", option=3
    ),
    "tokenpart_retr": _RetrievalMechanism(
        "token_parts", "learned-part-token-aggregation", option=6
    ),
    "transformer_fgret": _RetrievalMechanism(
        "transformer", "fine-grained-token-transformer"
    ),
}


def _build_pool(
    mechanism: _RetrievalMechanism,
    channels: int,
    depth: int,
) -> nn.Module:
    if mechanism.pooling == "average":
        return _AveragePool(channels)
    if mechanism.pooling == "gem":
        return _GeMPool(channels, initial_p=float(mechanism.option))
    if mechanism.pooling == "attention":
        return _AttentionPool(channels)
    if mechanism.pooling == "local_global":
        return _LocalGlobalPool(channels)
    if mechanism.pooling == "regional":
        return _RegionalPool(channels, grid=mechanism.option)
    if mechanism.pooling == "multiscale":
        return _MultiScalePool(channels)
    if mechanism.pooling == "vlad":
        return _NetVLADPool(channels, clusters=mechanism.option)
    if mechanism.pooling == "transformer":
        return _TransformerPool(channels, depth=depth)
    if mechanism.pooling == "scan":
        return _SelectiveScanPool(channels, depth=depth)
    if mechanism.pooling == "bilinear":
        return _BilinearPool(channels)
    if mechanism.pooling == "mixed":
        return _MixedPool(channels)
    if mechanism.pooling == "token_parts":
        return _TokenPartPool(channels, parts=mechanism.option)
    raise KeyError(f"Unknown retrieval pooling mechanism: {mechanism.pooling!r}")


class CompactRetrievalModel(nn.Module):
    def __init__(
        self,
        *,
        family: str,
        in_channels: int,
        width: int,
        depth: int,
        embed_dim: int,
    ) -> None:
        super().__init__()
        try:
            specification = _MECHANISMS[str(family)]
        except KeyError as error:
            raise KeyError(f"Unknown retrieval mechanism family: {family!r}") from error
        self.family = str(family)
        self.mechanism = specification.mechanism
        self.scoring = specification.scoring
        self.uses_context = specification.context
        self.encoder = TinyRetrievalEncoder(
            in_channels=int(in_channels), width=int(width), depth=int(depth)
        )
        self.pool = _build_pool(specification, self.encoder.out_channels, int(depth))
        self.projection = nn.Linear(int(self.pool.out_dim), int(embed_dim))
        if self.uses_context:
            self.learned_context = nn.Parameter(torch.zeros(1, int(embed_dim)))
            self.context_projection = nn.Linear(int(embed_dim), int(embed_dim))
        else:
            self.register_parameter("learned_context", None)
            self.context_projection = None
        if self.scoring == "pairwise":
            self.pair_scorer = nn.Sequential(
                nn.Linear(int(embed_dim) * 2, int(embed_dim)),
                nn.GELU(),
                nn.Linear(int(embed_dim), 1),
            )
        else:
            self.pair_scorer = None
        if self.scoring == "proxy":
            self.proxies = nn.Parameter(torch.randn(8, int(embed_dim)) * 0.02)
        else:
            self.register_parameter("proxies", None)
        self.logit_scale = nn.Parameter(torch.tensor(0.0))

    def _encode(
        self,
        image: torch.Tensor,
        context: torch.Tensor | None,
    ) -> torch.Tensor:
        descriptor = self.pool(self.encoder.forward_features(image))
        embedding = self.projection(descriptor)
        if self.uses_context:
            assert self.context_projection is not None
            if context is None:
                assert self.learned_context is not None
                context = self.learned_context.expand(embedding.shape[0], -1)
            if context.shape != embedding.shape:
                raise ValueError(
                    f"context must have shape {tuple(embedding.shape)}, got {tuple(context.shape)}"
                )
            embedding = embedding + 0.25 * torch.tanh(
                self.context_projection(context.to(embedding))
            )
        embedding = F.normalize(embedding, dim=-1)
        if self.proxies is not None:
            proxies = F.normalize(self.proxies, dim=-1)
            weights = torch.softmax(embedding @ proxies.t(), dim=-1)
            embedding = F.normalize(embedding + 0.2 * (weights @ proxies), dim=-1)
        return embedding

    def _similarity(
        self,
        query: torch.Tensor,
        gallery: torch.Tensor,
    ) -> torch.Tensor:
        cosine = query @ gallery.t()
        if self.scoring == "temperature":
            return cosine * self.logit_scale.exp().clamp(max=100.0)
        if self.scoring == "angular":
            sine = (1.0 - cosine.square()).clamp_min(1e-6).sqrt()
            return cosine - 0.1 * sine
        if self.scoring == "pairwise":
            assert self.pair_scorer is not None
            difference = (query[:, None] - gallery[None]).abs()
            interaction = query[:, None] * gallery[None]
            correction = self.pair_scorer(
                torch.cat((difference, interaction), dim=-1)
            ).squeeze(-1)
            return cosine + 0.1 * correction
        return cosine

    def forward(
        self,
        image: torch.Tensor,
        gallery: torch.Tensor | None = None,
        *,
        context: torch.Tensor | None = None,
        gallery_context: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        query = self._encode(image, context)
        output = {"embedding": query}
        if gallery is not None:
            gallery_embedding = self._encode(gallery, gallery_context)
            output["gallery_embedding"] = gallery_embedding
            output["similarity"] = self._similarity(query, gallery_embedding)
        return output


def build_compact_retrieval_model(
    *,
    family: str,
    variants: dict[str, dict[str, int]],
    in_channels: int,
    variant: str,
    width_mult: float = 1.0,
) -> nn.Module:
    if variant not in variants:
        raise KeyError(f"Unknown {family} retrieval variant: {variant!r}")
    spec = variants[variant]
    width = max(16, int(int(spec["width"]) * float(width_mult)))
    embed = max(64, int(int(spec["embed"]) * float(width_mult)))
    return CompactRetrievalModel(
        family=str(family),
        in_channels=int(in_channels),
        width=width,
        depth=int(spec["depth"]),
        embed_dim=embed,
    )


def build_baseline_retrieval_model(**kwargs: object) -> nn.Module:
    """Compatibility alias for callers migrating to the mechanism-aware builder."""

    return build_compact_retrieval_model(**kwargs)


def smoke_test_retrieval(builder, variant: str) -> None:
    model = builder(in_channels=3, variant=variant, width_mult=0.5)
    x = torch.randn(2, 3, 64, 64)
    gallery = torch.randn(3, 3, 64, 64)
    output = model(x, gallery)
    print(variant, model.mechanism, {key: tuple(value.shape) for key, value in output.items()})


__all__ = [
    "CompactRetrievalModel",
    "TinyRetrievalEncoder",
    "build_baseline_retrieval_model",
    "build_compact_retrieval_model",
    "check_nchw",
    "smoke_test_retrieval",
]
