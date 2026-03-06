from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from dlhub.vision.backbones._blocks import ConvBNAct, scale_channels


def check_nchw(x: torch.Tensor) -> torch.Tensor:
    x = x.to(torch.float32)
    if x.ndim != 4:
        raise ValueError(f"Expected input shape (B, C, H, W), got {tuple(x.shape)}")
    return x


_GROUP_SPECS: dict[str, dict[str, dict[str, int]]] = {
    "bilinear": {
        "tiny": {"stem": 24, "c2": 40, "c3": 64, "c4": 96, "embed": 96, "parts": 4, "depth": 1},
        "small": {"stem": 24, "c2": 48, "c3": 80, "c4": 128, "embed": 128, "parts": 6, "depth": 2},
        "base": {"stem": 32, "c2": 64, "c3": 96, "c4": 160, "embed": 160, "parts": 8, "depth": 3},
    },
    "part": {
        "tiny": {"stem": 24, "c2": 40, "c3": 64, "c4": 96, "embed": 96, "parts": 4, "depth": 1, "glimpses": 2},
        "small": {"stem": 24, "c2": 48, "c3": 80, "c4": 128, "embed": 128, "parts": 6, "depth": 2, "glimpses": 3},
        "base": {"stem": 32, "c2": 64, "c3": 96, "c4": 160, "embed": 160, "parts": 8, "depth": 3, "glimpses": 4},
    },
    "relation": {
        "tiny": {"stem": 24, "c2": 40, "c3": 64, "c4": 96, "embed": 96, "parts": 4, "depth": 1, "slots": 4},
        "small": {"stem": 24, "c2": 48, "c3": 80, "c4": 128, "embed": 128, "parts": 6, "depth": 2, "slots": 6},
        "base": {"stem": 32, "c2": 64, "c3": 96, "c4": 160, "embed": 160, "parts": 8, "depth": 3, "slots": 8},
    },
    "transformer": {
        "tiny": {"patch": 8, "embed": 96, "depth": 2, "heads": 4, "parts": 4},
        "small": {"patch": 8, "embed": 128, "depth": 3, "heads": 4, "parts": 6},
        "base": {"patch": 4, "embed": 160, "depth": 4, "heads": 8, "parts": 8},
    },
}


def make_fgvc_variants(prefix: str, *, group: str) -> dict[str, dict[str, int]]:
    if group not in _GROUP_SPECS:
        raise ValueError(f"Unknown FGVC group: {group!r}")
    return {f"{prefix}_{size}": dict(spec) for size, spec in _GROUP_SPECS[group].items()}


class ConvStage(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, *, depth: int = 1, stride: int = 2) -> None:
        super().__init__()
        d = int(depth)
        if d <= 0:
            raise ValueError("depth must be > 0")
        layers: list[nn.Module] = [ConvBNAct(int(in_ch), int(out_ch), kernel_size=3, stride=int(stride), act="relu")]
        for _ in range(d):
            layers.append(ConvBNAct(int(out_ch), int(out_ch), kernel_size=3, stride=1, act="relu"))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TinyFGBackbone(nn.Module):
    """Compact CNN backbone that returns /4, /8, /16 feature maps."""

    def __init__(self, *, in_channels: int, stem: int, c2: int, c3: int, c4: int, depth: int) -> None:
        super().__init__()
        self.stem = ConvBNAct(int(in_channels), int(stem), kernel_size=3, stride=2, act="relu")
        self.stage2 = ConvStage(int(stem), int(c2), depth=int(depth), stride=2)
        self.stage3 = ConvStage(int(c2), int(c3), depth=int(depth), stride=2)
        self.stage4 = ConvStage(int(c3), int(c4), depth=int(depth), stride=2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        c2 = self.stage2(x)
        c3 = self.stage3(c2)
        c4 = self.stage4(c3)
        return c2, c3, c4


class PartAttentionPool(nn.Module):
    def __init__(self, in_ch: int, num_parts: int) -> None:
        super().__init__()
        self.attn_head = nn.Conv2d(int(in_ch), int(num_parts), kernel_size=1)
        self.num_parts = int(num_parts)

    def forward(self, feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, c, h, w = feat.shape
        attn_logits = self.attn_head(feat)
        attn = attn_logits.flatten(2).softmax(dim=-1).view(b, self.num_parts, h, w)
        part_tokens = torch.einsum("bphw,bchw->bpc", attn, feat)
        return attn, part_tokens


class TinyPatchEncoder(nn.Module):
    def __init__(self, *, in_channels: int, image_size: int, patch_size: int, embed_dim: int, depth: int, heads: int) -> None:
        super().__init__()
        img = int(image_size)
        patch = int(patch_size)
        if img % patch != 0:
            raise ValueError(f"image_size ({img}) must be divisible by patch_size ({patch})")
        self.patch_embed = nn.Conv2d(int(in_channels), int(embed_dim), kernel_size=patch, stride=patch)
        grid = img // patch
        num_patches = grid * grid
        self.cls_token = nn.Parameter(torch.zeros(1, 1, int(embed_dim)))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, int(embed_dim)))
        layer = nn.TransformerEncoderLayer(
            d_model=int(embed_dim),
            nhead=int(heads),
            dim_feedforward=max(int(embed_dim) * 2, 64),
            dropout=0.0,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=int(depth))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed[:, : x.shape[1]]
        return self.encoder(x)


def _signed_sqrt_l2(x: torch.Tensor) -> torch.Tensor:
    x = torch.sign(x) * torch.sqrt(x.abs() + 1e-6)
    return F.normalize(x, dim=-1)


class BilinearFGVCModel(nn.Module):
    def __init__(
        self,
        *,
        family: str,
        spec: dict[str, int],
        in_channels: int,
        num_classes: int,
        image_size: int,
        width_mult: float,
        dropout: float,
    ) -> None:
        del image_size
        super().__init__()
        stem = scale_channels(int(spec["stem"]), float(width_mult), min_ch=16, divisor=8)
        c2 = scale_channels(int(spec["c2"]), float(width_mult), min_ch=16, divisor=8)
        c3 = scale_channels(int(spec["c3"]), float(width_mult), min_ch=16, divisor=8)
        c4 = scale_channels(int(spec["c4"]), float(width_mult), min_ch=16, divisor=8)
        embed = scale_channels(int(spec["embed"]), float(width_mult), min_ch=16, divisor=8)
        parts = int(spec["parts"])
        self.family = str(family)
        self.backbone = TinyFGBackbone(in_channels=int(in_channels), stem=stem, c2=c2, c3=c3, c4=c4, depth=int(spec["depth"]))
        self.part_pool = PartAttentionPool(c3, parts)
        self.global_proj = nn.Linear(c4, embed)
        self.part_proj = nn.Linear(c3, embed)
        self.hier_proj = nn.Linear(c2 + c3 + c4, embed)
        self.dropout = nn.Dropout(float(dropout))
        self.classifier = nn.Linear(embed, int(num_classes))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = check_nchw(x)
        c2, c3, c4 = self.backbone(x)
        attn, part_tokens = self.part_pool(c3)
        g = self.global_proj(F.adaptive_avg_pool2d(c4, (1, 1)).flatten(1))
        p = self.part_proj(part_tokens.mean(dim=1))
        h = self.hier_proj(
            torch.cat(
                [
                    F.adaptive_avg_pool2d(c2, (1, 1)).flatten(1),
                    F.adaptive_avg_pool2d(c3, (1, 1)).flatten(1),
                    F.adaptive_avg_pool2d(c4, (1, 1)).flatten(1),
                ],
                dim=-1,
            )
        )
        bilinear = g * p
        embedding = _signed_sqrt_l2(g + p + 0.5 * h + bilinear)
        logits = self.classifier(self.dropout(embedding))
        out = {
            "logits": logits,
            "embedding": embedding,
            "part_attn": attn,
            "part_tokens": part_tokens,
        }
        if self.family == "compact_bilinear":
            out["compact_sketch"] = torch.sin(g + p)
        elif self.family == "kernel_pooling":
            out["kernel_response"] = torch.relu(bilinear)
        elif self.family == "lowrank_bilinear":
            out["lowrank_factors"] = torch.stack([g, p], dim=1)
        elif self.family == "hierarchical_bilinear":
            out["hierarchical_embedding"] = h
        elif self.family == "isqrt_cov":
            out["covariance_descriptor"] = _signed_sqrt_l2(bilinear)
        elif self.family == "mpn_cov":
            out["matrix_power_descriptor"] = (g + p).pow(2)
        elif self.family == "ws_ban":
            out["attention_pooling"] = attn.flatten(2).mean(dim=-1)
        else:
            out["bilinear_descriptor"] = bilinear
        return out


class PartFGVCModel(nn.Module):
    def __init__(
        self,
        *,
        family: str,
        spec: dict[str, int],
        in_channels: int,
        num_classes: int,
        image_size: int,
        width_mult: float,
        dropout: float,
    ) -> None:
        del image_size
        super().__init__()
        stem = scale_channels(int(spec["stem"]), float(width_mult), min_ch=16, divisor=8)
        c2 = scale_channels(int(spec["c2"]), float(width_mult), min_ch=16, divisor=8)
        c3 = scale_channels(int(spec["c3"]), float(width_mult), min_ch=16, divisor=8)
        c4 = scale_channels(int(spec["c4"]), float(width_mult), min_ch=16, divisor=8)
        embed = scale_channels(int(spec["embed"]), float(width_mult), min_ch=16, divisor=8)
        parts = int(spec["parts"])
        glimpses = int(spec["glimpses"])
        self.family = str(family)
        self.backbone = TinyFGBackbone(in_channels=int(in_channels), stem=stem, c2=c2, c3=c3, c4=c4, depth=int(spec["depth"]))
        self.part_pool = PartAttentionPool(c3, parts)
        self.global_proj = nn.Linear(c4, embed)
        self.part_proj = nn.Linear(c3, embed)
        self.classifier = nn.Linear(embed, int(num_classes))
        self.box_head = nn.Linear(c3, 4)
        self.glimpse_head = nn.Linear(c3, glimpses)
        self.filter_head = nn.Linear(c3, parts)
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = check_nchw(x)
        c2, c3, c4 = self.backbone(x)
        attn, part_tokens = self.part_pool(c3)
        global_feat = self.global_proj(F.adaptive_avg_pool2d(c4, (1, 1)).flatten(1))
        part_feat = self.part_proj(part_tokens.mean(dim=1))
        embedding = torch.tanh(global_feat + part_feat)
        logits = self.classifier(self.dropout(embedding))
        proposal_boxes = torch.sigmoid(self.box_head(part_tokens))
        glimpse_logits = self.glimpse_head(part_tokens)
        out = {
            "logits": logits,
            "embedding": embedding,
            "part_attn": attn,
            "part_tokens": part_tokens,
            "proposal_boxes": proposal_boxes,
        }
        if self.family in {"part_rcnn", "partnet"}:
            out["proposal_scores"] = proposal_boxes.mean(dim=-1)
        elif self.family == "part_stacked_cnn":
            out["stacked_parts"] = part_tokens
        elif self.family == "pa_cnn":
            out["part_attention_logits"] = attn.mean(dim=(-2, -1))
        elif self.family == "racnn":
            out["glimpse_logits"] = glimpse_logits
        elif self.family == "ma_cnn":
            out["multi_attention_logits"] = glimpse_logits
        elif self.family == "dfl_cnn":
            out["filter_logits"] = self.filter_head(part_tokens)
        elif self.family == "nts_net":
            out["navigator_logits"] = glimpse_logits
        elif self.family == "tasn":
            out["sampling_logits"] = attn.flatten(2).mean(dim=-1)
        elif self.family == "s3n":
            out["snapshot_logits"] = c2.mean(dim=(2, 3))
        elif self.family == "mge_cnn":
            out["granularity_logits"] = torch.stack([part_tokens.mean(dim=1), part_tokens.max(dim=1).values], dim=1)
        elif self.family == "pmg":
            out["progressive_logits"] = torch.stack([global_feat, part_feat], dim=1)
        return out


class RelationFGVCModel(nn.Module):
    def __init__(
        self,
        *,
        family: str,
        spec: dict[str, int],
        in_channels: int,
        num_classes: int,
        image_size: int,
        width_mult: float,
        dropout: float,
    ) -> None:
        del image_size
        super().__init__()
        stem = scale_channels(int(spec["stem"]), float(width_mult), min_ch=16, divisor=8)
        c2 = scale_channels(int(spec["c2"]), float(width_mult), min_ch=16, divisor=8)
        c3 = scale_channels(int(spec["c3"]), float(width_mult), min_ch=16, divisor=8)
        c4 = scale_channels(int(spec["c4"]), float(width_mult), min_ch=16, divisor=8)
        embed = scale_channels(int(spec["embed"]), float(width_mult), min_ch=16, divisor=8)
        parts = int(spec["parts"])
        slots = int(spec["slots"])
        self.family = str(family)
        self.backbone = TinyFGBackbone(in_channels=int(in_channels), stem=stem, c2=c2, c3=c3, c4=c4, depth=int(spec["depth"]))
        self.part_pool = PartAttentionPool(c3, parts)
        self.global_proj = nn.Linear(c4, embed)
        self.part_proj = nn.Linear(c3, embed)
        self.prototype_bank = nn.Parameter(torch.randn(slots, embed) * 0.02)
        self.classifier = nn.Linear(embed, int(num_classes))
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = check_nchw(x)
        c2, c3, c4 = self.backbone(x)
        attn, part_tokens = self.part_pool(c3)
        global_feat = self.global_proj(F.adaptive_avg_pool2d(c4, (1, 1)).flatten(1))
        part_embed = self.part_proj(part_tokens)
        relation = torch.matmul(part_embed, part_embed.transpose(1, 2)) / math.sqrt(max(part_embed.shape[-1], 1))
        proto_scores = torch.einsum("bpc,kc->bpk", part_embed, self.prototype_bank).mean(dim=1)
        embedding = torch.tanh(global_feat + part_embed.mean(dim=1))
        logits = self.classifier(self.dropout(embedding))
        out = {
            "logits": logits,
            "embedding": embedding,
            "part_attn": attn,
            "relation_logits": relation,
            "prototype_scores": proto_scores,
        }
        if self.family == "osme_mamc":
            out["mutual_attention_logits"] = relation
        elif self.family == "api_net":
            out["pair_interaction_logits"] = relation.mean(dim=-1)
        elif self.family == "crossx":
            out["complementary_logits"] = part_embed.sum(dim=1)
        elif self.family == "region_grouping":
            out["region_groups"] = attn.flatten(2).argmax(dim=-1)
        elif self.family == "dcl":
            out["destruct_logits"] = -part_embed.mean(dim=1)
            out["construct_logits"] = part_embed.mean(dim=1)
        elif self.family == "ws_dan":
            out["augmented_attn"] = attn
        elif self.family == "proto_pnet":
            out["prototype_activations"] = proto_scores
        elif self.family == "hse":
            out["hierarchy_logits"] = torch.stack([global_feat, part_embed.mean(dim=1)], dim=1)
        elif self.family == "interp_parts":
            out["interpretable_part_scores"] = attn.mean(dim=(-2, -1))
        elif self.family == "ga_cnn":
            out["granularity_alignment"] = torch.cat([global_feat, part_embed.mean(dim=1)], dim=-1)
        return out


class TransformerFGVCModel(nn.Module):
    def __init__(
        self,
        *,
        family: str,
        spec: dict[str, int],
        in_channels: int,
        num_classes: int,
        image_size: int,
        width_mult: float,
        dropout: float,
    ) -> None:
        super().__init__()
        embed = scale_channels(int(spec["embed"]), float(width_mult), min_ch=32, divisor=8)
        self.family = str(family)
        self.encoder = TinyPatchEncoder(
            in_channels=int(in_channels),
            image_size=int(image_size),
            patch_size=int(spec["patch"]),
            embed_dim=int(embed),
            depth=int(spec["depth"]),
            heads=int(spec["heads"]),
        )
        self.token_scorer = nn.Linear(int(embed), 1)
        self.meta_token = nn.Parameter(torch.randn(1, 1, int(embed)) * 0.02)
        self.semantic_token = nn.Parameter(torch.randn(1, 1, int(embed)) * 0.02)
        self.classifier = nn.Linear(int(embed), int(num_classes))
        self.dropout = nn.Dropout(float(dropout))
        self.num_parts = int(spec["parts"])

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = check_nchw(x)
        tokens = self.encoder(x)
        cls_token = tokens[:, 0]
        patch_tokens = tokens[:, 1:]
        scores = self.token_scorer(patch_tokens).squeeze(-1)
        k = min(self.num_parts, patch_tokens.shape[1])
        values, indices = torch.topk(scores, k=k, dim=1)
        gather_idx = indices.unsqueeze(-1).expand(-1, -1, patch_tokens.shape[-1])
        selected = torch.gather(patch_tokens, 1, gather_idx)
        pooled = selected.mean(dim=1)
        embedding = torch.tanh(cls_token + pooled)
        logits = self.classifier(self.dropout(embedding))
        out = {
            "logits": logits,
            "embedding": embedding,
            "selected_indices": indices,
            "selected_scores": values,
            "selected_tokens": selected,
        }
        if self.family == "transfg":
            out["part_tokens"] = selected
        elif self.family == "ffvt":
            out["fusion_token"] = pooled
        elif self.family == "pedtrans":
            out["pose_token"] = cls_token + self.meta_token.squeeze(1)
        elif self.family == "vit_fod":
            out["difference_token"] = cls_token - pooled
        elif self.family == "aftrans":
            out["fused_attention"] = scores
        elif self.family == "sim_trans":
            out["similarity_logits"] = torch.matmul(selected, selected.transpose(1, 2))
        elif self.family == "pca_net":
            out["co_attention"] = torch.softmax(torch.matmul(selected, selected.transpose(1, 2)), dim=-1)
        elif self.family == "metaformer_fgvc":
            out["meta_token"] = cls_token + self.meta_token.squeeze(1)
        elif self.family == "pim":
            out["plugin_mask"] = scores
        elif self.family == "cvl":
            out["language_token"] = cls_token + self.semantic_token.squeeze(1)
        return out


def smoke_test_classifier(builder, variant: str) -> None:
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    model = builder(in_channels=3, num_classes=5, variant=variant, image_size=64, width_mult=0.5, dropout=0.0)
    out = model(x)
    print(variant, {k: tuple(v.shape) for k, v in out.items() if torch.is_tensor(v)})
    loss = sum(v.to(torch.float32).mean() for v in out.values() if torch.is_tensor(v))
    loss.backward()
    print("ok")


def build_fgvc_model(
    cls,
    *,
    variants: dict[str, dict[str, int]],
    in_channels: int,
    num_classes: int,
    variant: str,
    image_size: int = 64,
    width_mult: float = 1.0,
    dropout: float = 0.1,
    family: str,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in variants:
        raise ValueError(f"Unknown {family} variant: {variant!r}. Supported: {sorted(variants)}")
    return cls(
        family=str(family),
        spec=dict(variants[name]),
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        image_size=int(image_size),
        width_mult=float(width_mult),
        dropout=float(dropout),
    )


def build_zoo_model(builder_name: str, cfg: Any) -> nn.Module:
    kwargs = {
        "in_channels": int(cfg.in_channels),
        "num_classes": int(cfg.num_classes),
        "image_size": int(cfg.image_size),
        "width_mult": float(cfg.width_mult),
        "dropout": float(cfg.dropout),
    }
    return builder_name(**kwargs)
