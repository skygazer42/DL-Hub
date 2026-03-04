from __future__ import annotations

import torch
from torch import nn

from dlhub.nlp.algorithms._helpers import make_builder
from dlhub.nlp.types import Builder

from ._transformer_core import TransformerConfig, TransformerTextClassifier


class TransformerClassifier(TransformerTextClassifier):
    pass


def build_transformer_classifier(
    *,
    vocab_size: int,
    pad_id: int,
    max_length: int,
    num_classes: int,
    width_mult: float = 1.0,
    dropout: float = 0.1,
    variant: str,
) -> nn.Module:
    name = str(variant).lower().strip()

    if name in {"transformer_tiny", "transformer"}:
        embed_dim, heads, layers = 192, 4, 2
        pos, rope, alibi, relb = "learned", False, False, False
        ffn, norm, prenorm, pool = "relu", "layer", True, "mean"
        kv_heads = None
        attn_impl = "full"
        linformer_k = 32
    elif name in {"transformer_small"}:
        embed_dim, heads, layers = 256, 4, 3
        pos, rope, alibi, relb = "learned", False, False, False
        ffn, norm, prenorm, pool = "relu", "layer", True, "mean"
        kv_heads = None
        attn_impl = "full"
        linformer_k = 32
    elif name in {"transformer_base"}:
        embed_dim, heads, layers = 320, 5, 4
        pos, rope, alibi, relb = "learned", False, False, False
        ffn, norm, prenorm, pool = "gelu", "layer", True, "mean"
        kv_heads = None
        attn_impl = "full"
        linformer_k = 32
    elif name in {"transformer_rope_tiny"}:
        embed_dim, heads, layers = 192, 4, 2
        pos, rope, alibi, relb = "none", True, False, False
        ffn, norm, prenorm, pool = "gelu", "layer", True, "mean"
        kv_heads = None
        attn_impl = "full"
        linformer_k = 32
    elif name in {"transformer_alibi_tiny"}:
        embed_dim, heads, layers = 192, 4, 2
        pos, rope, alibi, relb = "none", False, True, False
        ffn, norm, prenorm, pool = "gelu", "layer", True, "mean"
        kv_heads = None
        attn_impl = "full"
        linformer_k = 32
    elif name in {"transformer_relbias_tiny"}:
        embed_dim, heads, layers = 192, 4, 2
        pos, rope, alibi, relb = "none", False, False, True
        ffn, norm, prenorm, pool = "gelu", "layer", True, "mean"
        kv_heads = None
        attn_impl = "full"
        linformer_k = 32
    elif name in {"transformer_mqa_tiny"}:
        embed_dim, heads, layers = 192, 4, 2
        pos, rope, alibi, relb = "learned", False, False, False
        ffn, norm, prenorm, pool = "gelu", "layer", True, "mean"
        kv_heads = 1
        attn_impl = "full"
        linformer_k = 32
    elif name in {"transformer_gqa_tiny"}:
        embed_dim, heads, layers = 192, 4, 2
        pos, rope, alibi, relb = "learned", False, False, False
        ffn, norm, prenorm, pool = "gelu", "layer", True, "mean"
        kv_heads = 2
        attn_impl = "full"
        linformer_k = 32
    else:
        # --- Structured "lab" variants (generated in local_zoo):
        # Format (full attention):
        #   transformer_{size}_{pool}_{ffn}_{norm}_{pos}_{prepost}
        #
        # Format (Performer attention):
        #   transformer_performer_{size}_{pool}_{ffn}_{norm}_{pos}_{prepost}
        #
        # Format (Linformer attention):
        #   transformer_linformer_{size}_k{K}_{pool}_{ffn}_{norm}_{pos}_{prepost}
        # Examples:
        #   transformer_tiny_mean_gelu_ln_learned_pre
        #   transformer_small_cls_swiglu_rms_sin_post
        parts = name.split("_")
        if len(parts) == 7 and parts[0] == "transformer":
            attn_impl = "full"
            linformer_k = 32
            _, size, pool, ffn, norm_token, pos_token, prepost = parts
        elif len(parts) == 8 and parts[0] == "transformer":
            _, impl, size, pool, ffn, norm_token, pos_token, prepost = parts
            if impl == "performer":
                attn_impl = "performer"
                linformer_k = 32
            else:
                raise ValueError(f"Unknown transformer impl: {impl!r}")
        elif len(parts) == 9 and parts[0] == "transformer":
            _, impl, size, k_token, pool, ffn, norm_token, pos_token, prepost = parts
            if impl == "linformer":
                if not k_token.startswith("k") or len(k_token) == 1:
                    raise ValueError("Expected Linformer token like 'k16'")
                linformer_k = int(k_token.removeprefix("k"))
                if linformer_k <= 0:
                    raise ValueError("linformer_k must be > 0")
                attn_impl = "linformer"
            else:
                raise ValueError(f"Unknown transformer impl: {impl!r}")
        else:
            raise ValueError(
                "Unknown transformer variant. Supported: transformer_tiny|transformer_small|transformer_base|"
                "transformer_rope_tiny|transformer_alibi_tiny|transformer_relbias_tiny|"
                "transformer_mqa_tiny|transformer_gqa_tiny|"
                "transformer_{size}_{pool}_{ffn}_{norm}_{pos}_{prepost}|"
                "transformer_performer_{size}_{pool}_{ffn}_{norm}_{pos}_{prepost}|"
                "transformer_linformer_{size}_k{K}_{pool}_{ffn}_{norm}_{pos}_{prepost}"
            )

        if size == "tiny":
            embed_dim, heads, layers = 192, 4, 2
        elif size == "small":
            embed_dim, heads, layers = 256, 4, 3
        elif size == "base":
            embed_dim, heads, layers = 320, 5, 4
        else:
            raise ValueError(f"Unknown transformer size: {size!r}")

        pool = str(pool).lower().strip()
        if pool not in {"mean", "cls", "attn"}:
            raise ValueError("pool must be one of: mean|cls|attn")

        ffn = str(ffn).lower().strip()
        if ffn not in {"relu", "gelu", "swiglu", "geglu"}:
            raise ValueError("ffn must be one of: relu|gelu|swiglu|geglu")

        norm_token = str(norm_token).lower().strip()
        if norm_token in {"ln", "layer", "layernorm"}:
            norm = "layer"
        elif norm_token in {"rms", "rmsnorm"}:
            norm = "rms"
        else:
            raise ValueError("norm must be one of: ln|rms")

        pos_token = str(pos_token).lower().strip()
        if pos_token not in {"learned", "sin", "none"}:
            raise ValueError("pos must be one of: learned|sin|none")
        pos = pos_token

        prepost = str(prepost).lower().strip()
        if prepost in {"pre", "prenorm"}:
            prenorm = True
        elif prepost in {"post", "postnorm"}:
            prenorm = False
        else:
            raise ValueError("prepost must be one of: pre|post")

        rope, alibi, relb = False, False, False
        kv_heads = None

    return TransformerClassifier(
        TransformerConfig(
            vocab_size=int(vocab_size),
            pad_id=int(pad_id),
            max_length=int(max_length),
            num_classes=int(num_classes),
            width_mult=float(width_mult),
            dropout=float(dropout),
            embed_dim=int(embed_dim),
            num_heads=int(heads),
            num_layers=int(layers),
            pos=str(pos),
            rope=bool(rope),
            alibi=bool(alibi),
            rel_bias=bool(relb),
            attn_impl=str(attn_impl),
            num_kv_heads=kv_heads,
            linformer_k=int(linformer_k),
            longformer_window=8,
            ffn_kind=str(ffn),
            norm_kind=str(norm),
            prenorm=bool(prenorm),
            causal=False,
            pool=str(pool),
            share_layers=False,
        )
    )


def registry() -> dict[str, Builder]:
    r: dict[str, Builder] = {}

    # Family alias (historically `nl:transformer`)
    r["transformer"] = make_builder(build_transformer_classifier, variant="transformer_tiny")

    # Base named variants
    for name in (
        "transformer_tiny",
        "transformer_small",
        "transformer_base",
        "transformer_rope_tiny",
        "transformer_alibi_tiny",
        "transformer_relbias_tiny",
        "transformer_mqa_tiny",
        "transformer_gqa_tiny",
    ):
        r[name] = make_builder(build_transformer_classifier, variant=name)

    # --- Structured "lab" variants (mirrors parsing in `build_transformer_classifier`)
    sizes = ("tiny", "small")
    pools = ("mean", "cls")
    ffns_full = ("gelu", "swiglu", "geglu")
    ffns_perf = ("gelu", "swiglu")
    norms = ("ln", "rms")
    poss = ("learned", "sin")
    preposts = ("pre", "post")

    # Full attention: all combos for mean/cls pools.
    for size in sizes:
        for pool in pools:
            for ffn in ffns_full:
                for norm in norms:
                    for pos in poss:
                        for prepost in preposts:
                            name = f"transformer_{size}_{pool}_{ffn}_{norm}_{pos}_{prepost}"
                            r[name] = make_builder(build_transformer_classifier, variant=name)

    # A tiny curated set of attention-pooled variants (explicit arch ids).
    for name in (
        "transformer_tiny_attn_gelu_ln_learned_pre",
        "transformer_tiny_attn_swiglu_rms_learned_pre",
        "transformer_small_attn_geglu_ln_sin_pre",
        "transformer_small_attn_gelu_rms_sin_post",
    ):
        r[name] = make_builder(build_transformer_classifier, variant=name)

    # Performer: all (size, pool, ffn, norm, pos) combos with pre-norm,
    # plus a few explicit post-norm variants.
    for size in sizes:
        for pool in pools:
            for ffn in ffns_perf:
                for norm in norms:
                    for pos in poss:
                        name = f"transformer_performer_{size}_{pool}_{ffn}_{norm}_{pos}_pre"
                        r[name] = make_builder(build_transformer_classifier, variant=name)
    for name in (
        "transformer_performer_small_cls_gelu_rms_learned_post",
        "transformer_performer_small_mean_swiglu_ln_sin_post",
        "transformer_performer_tiny_cls_swiglu_rms_sin_post",
        "transformer_performer_tiny_mean_gelu_ln_learned_post",
    ):
        r[name] = make_builder(build_transformer_classifier, variant=name)

    # Linformer: all combos for (size, pool, ffn, norm, pos, pre/post) with fixed k.
    for size, k in (("tiny", 16), ("small", 24)):
        for pool in pools:
            for ffn in ffns_perf:
                for norm in norms:
                    for pos in poss:
                        for prepost in preposts:
                            name = f"transformer_linformer_{size}_k{k}_{pool}_{ffn}_{norm}_{pos}_{prepost}"
                            r[name] = make_builder(build_transformer_classifier, variant=name)

    return r


def _smoke() -> None:
    vocab_size = 128
    pad_id = 0
    max_length = 32
    num_classes = 4

    model = build_transformer_classifier(
        vocab_size=vocab_size,
        pad_id=pad_id,
        max_length=max_length,
        num_classes=num_classes,
        width_mult=0.5,
        dropout=0.1,
        variant="transformer_rope_tiny",
    )
    model.eval()

    x = torch.randint(0, vocab_size, (2, max_length), dtype=torch.long)
    attention_mask = torch.ones((2, max_length), dtype=torch.float32)
    with torch.no_grad():
        y = model({"input_ids": x, "attention_mask": attention_mask})

    n_params = sum(int(p.numel()) for p in model.parameters())
    print(f"smoke_ok: y.shape={tuple(y.shape)} params={n_params}")


if __name__ == "__main__":
    _smoke()


__all__ = ["TransformerClassifier", "build_transformer_classifier", "registry"]
