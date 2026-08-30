from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn
from torch.nn import functional as F


def _resolve_device(
    device: torch.device | str | None,
    *,
    fallback: torch.device,
) -> torch.device:
    if device is None:
        return fallback
    if isinstance(device, torch.device):
        return device
    return torch.device(device)


def _transformer(width: int, depth: int, dropout: float) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(
        d_model=width,
        nhead=4,
        dim_feedforward=width * 2,
        dropout=float(dropout),
        activation="gelu",
        batch_first=True,
        norm_first=False,
    )
    return nn.TransformerEncoder(
        layer,
        num_layers=max(1, int(depth)),
        enable_nested_tensor=False,
    )


class CompactVLM(nn.Module):
    """Compact multimodal core with explicit dual, stream, fusion, and bridge paths."""

    def __init__(
        self,
        *,
        family: str,
        image_size: int,
        vocab_size: int,
        seq_len: int,
        embed_dim: int,
        width: int,
        depth: int,
        num_classes: int = 0,
        dropout: float = 0.0,
        architecture_mode: str = "dual_encoder",
        use_instruction: bool = False,
        use_query_bridge: bool = False,
        use_generation_head: bool = False,
    ) -> None:
        super().__init__()
        self.family = str(family)
        self.image_size = int(image_size)
        self.vocab_size = int(vocab_size)
        self.seq_len = int(seq_len)
        self.embed_dim = int(embed_dim)
        self.num_classes = int(max(0, num_classes))
        self.architecture_mode = str(architecture_mode).strip().lower()
        self.use_instruction = bool(use_instruction)
        self.use_query_bridge = bool(use_query_bridge)
        self.use_generation_head = bool(use_generation_head)
        dropout_rate = float(dropout)
        if not 0.0 <= dropout_rate < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.architecture_mode not in {
            "dual_encoder",
            "single_stream",
            "fusion",
            "bridge",
        }:
            raise ValueError(f"Unsupported VLM architecture mode: {architecture_mode!r}")

        hidden = max(int(width), self.embed_dim)
        hidden = ((hidden + 3) // 4) * 4
        self.hidden = hidden
        self.mechanism = {
            "dual_encoder": "contrastive-dual-encoder",
            "single_stream": "joint-multimodal-transformer",
            "fusion": "text-to-image-cross-attention",
            "bridge": "query-token-vision-language-bridge",
        }[self.architecture_mode]

        self.image_patch_encoder = nn.Sequential(
            nn.Conv2d(3, hidden // 2, kernel_size=4, stride=4),
            nn.GELU(),
            nn.Conv2d(hidden // 2, hidden, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.image_norm = nn.LayerNorm(hidden)
        self.image_projection = nn.Linear(hidden, self.embed_dim)
        self.embedding_dropout = nn.Dropout(dropout_rate)

        self.token_embed = nn.Embedding(self.vocab_size, hidden)
        self.text_positions = nn.Parameter(torch.randn(1, self.seq_len, hidden) * 0.01)
        self.text_encoder = _transformer(hidden, depth, dropout_rate)
        self.text_projection = nn.Linear(hidden, self.embed_dim)

        self.instruction_projection: nn.Module | None
        if self.use_instruction:
            self.instruction_projection = nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Linear(hidden, hidden),
            )
        else:
            self.instruction_projection = None

        self.fusion_attention: nn.MultiheadAttention | None = None
        self.joint_encoder: nn.TransformerEncoder | None = None
        self.vision_bridge: nn.MultiheadAttention | None = None
        self.language_bridge: nn.MultiheadAttention | None = None
        self.register_parameter("stream_type_embeddings", None)
        self.register_parameter("query_tokens", None)

        if self.architecture_mode == "single_stream":
            self.stream_type_embeddings = nn.Parameter(torch.randn(2, hidden) * 0.01)
            self.joint_encoder = _transformer(hidden, depth, dropout_rate)
        elif self.architecture_mode == "fusion":
            self.fusion_attention = nn.MultiheadAttention(
                hidden,
                num_heads=4,
                dropout=dropout_rate,
                batch_first=True,
            )
        elif self.architecture_mode == "bridge":
            query_count = 4 if self.use_query_bridge else 1
            self.query_tokens = nn.Parameter(torch.randn(query_count, hidden) * 0.02)
            self.vision_bridge = nn.MultiheadAttention(
                hidden,
                num_heads=4,
                dropout=dropout_rate,
                batch_first=True,
            )
            self.language_bridge = nn.MultiheadAttention(
                hidden,
                num_heads=4,
                dropout=dropout_rate,
                batch_first=True,
            )

        self.joint_projection = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, self.embed_dim),
        )
        logits_dim = self.num_classes if self.num_classes > 0 else self.vocab_size
        self.classifier = nn.Linear(self.embed_dim, logits_dim)

        if self.use_generation_head:
            self.generation_context = nn.Linear(self.embed_dim, hidden)
            self.generation_norm = nn.LayerNorm(hidden)
            self.generation_head = nn.Linear(hidden, self.vocab_size)
        else:
            self.generation_context = None
            self.generation_norm = None
            self.generation_head = None
        self.last_cross_attention: torch.Tensor | None = None

    def _make_token_batch(
        self,
        *,
        batch_size: int,
        length: int,
        device: torch.device,
        ids: torch.Tensor | None,
    ) -> torch.Tensor:
        if ids is None:
            return torch.randint(
                0,
                self.vocab_size,
                (int(batch_size), int(length)),
                device=device,
            )
        if ids.ndim != 2:
            raise ValueError(f"token ids must be 2D, got shape {tuple(ids.shape)}")
        if int(ids.shape[0]) != int(batch_size):
            raise ValueError(
                f"token batch mismatch: expected {batch_size}, got {int(ids.shape[0])}"
            )
        if int(ids.shape[1]) > self.seq_len:
            raise ValueError(
                f"token length {int(ids.shape[1])} exceeds configured seq_len {self.seq_len}"
            )
        return ids.to(device=device, dtype=torch.long)

    def _encode_images(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"images must have shape (B, 3, H, W), got {tuple(images.shape)}")
        tokens = self.image_patch_encoder(images).flatten(2).transpose(1, 2)
        tokens = self.embedding_dropout(self.image_norm(tokens))
        embedding = self.image_projection(tokens.mean(dim=1))
        return tokens, embedding

    def _encode_text(self, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.embedding_dropout(self.token_embed(ids))
        tokens = tokens + self.text_positions[:, : tokens.shape[1]]
        tokens = self.text_encoder(tokens)
        embedding = self.text_projection(tokens.mean(dim=1))
        return tokens, embedding

    def _apply_instruction(
        self,
        text_tokens: torch.Tensor,
        instruction_ids: torch.Tensor | None,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        if not self.use_instruction:
            return text_tokens
        assert self.instruction_projection is not None
        instruction_ids = self._make_token_batch(
            batch_size=text_tokens.shape[0],
            length=max(4, self.seq_len // 2),
            device=device,
            ids=instruction_ids,
        )
        instruction = self.token_embed(instruction_ids).mean(dim=1)
        instruction = self.instruction_projection(instruction)
        return text_tokens + instruction[:, None]

    def _joint_state(
        self,
        image_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        image_embed: torch.Tensor,
        text_embed: torch.Tensor,
    ) -> torch.Tensor:
        if self.architecture_mode == "dual_encoder":
            return 0.5 * (image_embed + text_embed)
        if self.architecture_mode == "single_stream":
            assert self.joint_encoder is not None
            assert self.stream_type_embeddings is not None
            image_state = image_tokens + self.stream_type_embeddings[0]
            text_state = text_tokens + self.stream_type_embeddings[1]
            joint_tokens = self.joint_encoder(torch.cat((image_state, text_state), dim=1))
            return self.joint_projection(joint_tokens.mean(dim=1))
        if self.architecture_mode == "fusion":
            assert self.fusion_attention is not None
            fused, attention = self.fusion_attention(
                text_tokens,
                image_tokens,
                image_tokens,
                need_weights=True,
                average_attn_weights=False,
            )
            self.last_cross_attention = attention.detach()
            return self.joint_projection((text_tokens + fused).mean(dim=1))

        assert self.query_tokens is not None
        assert self.vision_bridge is not None
        assert self.language_bridge is not None
        queries = self.query_tokens[None].expand(image_tokens.shape[0], -1, -1)
        vision_queries, vision_attention = self.vision_bridge(
            queries,
            image_tokens,
            image_tokens,
            need_weights=True,
            average_attn_weights=False,
        )
        language_queries, language_attention = self.language_bridge(
            vision_queries,
            text_tokens,
            text_tokens,
            need_weights=True,
            average_attn_weights=False,
        )
        self.last_cross_attention = torch.cat(
            (
                vision_attention.flatten(2),
                language_attention.flatten(2),
            ),
            dim=-1,
        ).detach()
        return self.joint_projection((vision_queries + language_queries).mean(dim=1))

    def forward(
        self,
        *,
        batch_size: int = 2,
        device: torch.device | str | None = None,
        images: torch.Tensor | None = None,
        input_ids: torch.Tensor | None = None,
        instruction_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        parameter_device = next(self.parameters()).device
        target_device = _resolve_device(device, fallback=parameter_device)
        batch = int(batch_size)
        if images is not None:
            batch = int(images.shape[0])
        elif input_ids is not None:
            batch = int(input_ids.shape[0])

        if images is None:
            images = torch.randn(
                batch,
                3,
                self.image_size,
                self.image_size,
                device=target_device,
            )
        else:
            images = images.to(device=target_device, dtype=torch.float32)
        image_tokens, image_embed = self._encode_images(images)

        text_ids = self._make_token_batch(
            batch_size=batch,
            length=self.seq_len,
            device=target_device,
            ids=input_ids,
        )
        text_tokens, text_embed = self._encode_text(text_ids)
        text_tokens = self._apply_instruction(
            text_tokens,
            instruction_ids,
            device=target_device,
        )
        if self.use_instruction:
            text_embed = self.text_projection(text_tokens.mean(dim=1))

        joint = self._joint_state(
            image_tokens,
            text_tokens,
            image_embed,
            text_embed,
        )

        if self.architecture_mode == "dual_encoder":
            logits = F.normalize(image_embed, dim=1) @ F.normalize(text_embed, dim=1).t()
        else:
            logits = self.classifier(joint)

        output: dict[str, torch.Tensor] = {
            "image_embed": image_embed,
            "text_embed": text_embed,
            "logits": logits,
        }
        if self.generation_head is not None:
            assert self.generation_context is not None
            assert self.generation_norm is not None
            generation_states = text_tokens + self.generation_context(joint)[:, None]
            token_logits = self.generation_head(self.generation_norm(generation_states))
            output["token_logits"] = token_logits
            output["generated_tokens"] = token_logits.argmax(dim=-1)
        return output


def build_compact_vlm_family(
    *,
    family: str,
    variants: dict[str, dict[str, int]],
    image_size: int = 32,
    vocab_size: int = 128,
    seq_len: int = 16,
    embed_dim: int = 64,
    num_classes: int = 0,
    variant: str,
    width_mult: float = 1.0,
    dropout: float = 0.0,
    architecture_mode: str,
    use_instruction: bool = False,
    use_query_bridge: bool = False,
    use_generation_head: bool = False,
) -> nn.Module:
    if variant not in variants:
        raise KeyError(f"Unknown {family} VLM variant: {variant!r}")
    config = variants[variant]
    width = max(16, int(int(config["width"]) * float(width_mult)))
    dimension = max(int(embed_dim), int(config["embed"]))
    return CompactVLM(
        family=str(family),
        image_size=int(image_size),
        vocab_size=int(vocab_size),
        seq_len=int(seq_len),
        embed_dim=dimension,
        width=width,
        depth=int(config["depth"]),
        num_classes=int(num_classes),
        dropout=float(dropout),
        architecture_mode=str(architecture_mode),
        use_instruction=bool(use_instruction),
        use_query_bridge=bool(use_query_bridge),
        use_generation_head=bool(use_generation_head),
    )


def build_baseline_vlm_family(**kwargs: object) -> nn.Module:
    """Compatibility entrypoint for family labels still using shared mode baselines."""

    return build_compact_vlm_family(**kwargs)


def smoke_test_vlm(builder: Callable[..., nn.Module], variant: str) -> None:
    model = builder(
        image_size=32,
        vocab_size=128,
        seq_len=16,
        embed_dim=64,
        num_classes=8,
        variant=variant,
        width_mult=0.5,
        dropout=0.0,
    )
    output = model.forward(
        images=torch.randn(2, 3, 32, 32),
        input_ids=torch.randint(0, 128, (2, 16)),
    )
    shapes = {key: tuple(value.shape) for key, value in output.items()}
    print(variant, model.mechanism, shapes)


__all__ = [
    "CompactVLM",
    "build_baseline_vlm_family",
    "build_compact_vlm_family",
    "smoke_test_vlm",
]
