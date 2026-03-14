from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn


def _make_mlp(
    *,
    in_dim: int,
    out_dim: int,
    width: int,
    depth: int,
    dropout: float,
) -> nn.Module:
    layers: list[nn.Module] = []
    cur = int(in_dim)
    for _ in range(max(1, int(depth))):
        layers.append(nn.Linear(cur, int(width)))
        layers.append(nn.GELU())
        if float(dropout) > 0:
            layers.append(nn.Dropout(float(dropout)))
        cur = int(width)
    layers.append(nn.Linear(cur, int(out_dim)))
    return nn.Sequential(*layers)


def _resolve_device(device: torch.device | str | None) -> torch.device:
    if device is None:
        return torch.device("cpu")
    if isinstance(device, torch.device):
        return device
    return torch.device(device)


class ToyVLM(nn.Module):
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
        self.image_flat_dim = 3 * self.image_size * self.image_size
        hidden = max(int(width), self.embed_dim)

        self.image_encoder = _make_mlp(
            in_dim=self.image_flat_dim,
            out_dim=self.embed_dim,
            width=hidden,
            depth=int(depth),
            dropout=float(dropout),
        )
        self.token_embed = nn.Embedding(self.vocab_size, hidden)
        self.text_proj = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, self.embed_dim),
        )
        self.instruction_proj: nn.Module | None
        if self.use_instruction:
            self.instruction_proj = nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Linear(hidden, self.embed_dim),
            )
        else:
            self.instruction_proj = None

        self.fusion_proj = nn.Sequential(
            nn.Linear(self.embed_dim * 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, self.embed_dim),
        )
        self.bridge_proj: nn.Module | None
        if self.use_query_bridge:
            self.query_tokens = nn.Parameter(torch.randn(4, self.embed_dim) * 0.02)
            self.bridge_proj = nn.Sequential(
                nn.Linear(self.embed_dim * 2, hidden),
                nn.GELU(),
                nn.Linear(hidden, self.embed_dim),
            )
        else:
            self.register_parameter("query_tokens", None)
            self.bridge_proj = None

        logits_dim = self.num_classes if self.num_classes > 0 else self.vocab_size
        self.classifier = nn.Linear(self.embed_dim, logits_dim)
        self.generation_head: nn.Module | None
        if self.use_generation_head:
            self.generation_head = nn.Linear(self.embed_dim, self.vocab_size)
        else:
            self.generation_head = None

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
        return ids.to(device=device, dtype=torch.long)

    def _encode_text_tokens(self, ids: torch.Tensor, *, use_instruction: bool) -> torch.Tensor:
        pooled = self.token_embed(ids).mean(dim=1)
        if use_instruction:
            assert self.instruction_proj is not None
            return self.instruction_proj(pooled)
        return self.text_proj(pooled)

    def _joint_repr(
        self,
        image_embed: torch.Tensor,
        text_embed: torch.Tensor,
        *,
        instruction_embed: torch.Tensor | None,
    ) -> torch.Tensor:
        text_state = text_embed
        if instruction_embed is not None:
            text_state = text_state + instruction_embed

        mode = self.architecture_mode
        if mode == "dual_encoder":
            return 0.5 * (image_embed + text_state)
        if mode == "single_stream":
            return self.fusion_proj(torch.cat([image_embed, text_state], dim=1))
        if mode == "fusion":
            return self.fusion_proj(torch.cat([image_embed, text_state], dim=1))
        if mode == "bridge":
            if self.bridge_proj is None or self.query_tokens is None:
                return self.fusion_proj(torch.cat([image_embed, text_state], dim=1))
            query = self.query_tokens.unsqueeze(0).expand(image_embed.shape[0], -1, -1).mean(dim=1)
            bridge_state = image_embed + query
            return self.bridge_proj(torch.cat([bridge_state, text_state], dim=1))
        return self.fusion_proj(torch.cat([image_embed, text_state], dim=1))

    def forward(
        self,
        *,
        batch_size: int = 2,
        device: torch.device | str | None = None,
        input_ids: torch.Tensor | None = None,
        instruction_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        dev = _resolve_device(device)
        batch = int(batch_size)
        images = torch.randn(batch, 3, self.image_size, self.image_size, device=dev)
        image_embed = self.image_encoder(images.view(batch, -1))

        text_ids = self._make_token_batch(
            batch_size=batch,
            length=self.seq_len,
            device=dev,
            ids=input_ids,
        )
        text_embed = self._encode_text_tokens(text_ids, use_instruction=False)

        instruction_embed: torch.Tensor | None = None
        if self.use_instruction:
            instr_ids = self._make_token_batch(
                batch_size=batch,
                length=max(4, self.seq_len // 2),
                device=dev,
                ids=instruction_ids,
            )
            instruction_embed = self._encode_text_tokens(instr_ids, use_instruction=True)

        joint = self._joint_repr(
            image_embed,
            text_embed,
            instruction_embed=instruction_embed,
        )

        if self.architecture_mode == "dual_encoder":
            logits = F.normalize(image_embed, dim=1) @ F.normalize(text_embed, dim=1).t()
        else:
            logits = self.classifier(joint)

        out: dict[str, torch.Tensor] = {
            "image_embed": image_embed,
            "text_embed": text_embed,
            "logits": logits,
        }
        if self.generation_head is not None:
            token_logits = self.generation_head(joint).unsqueeze(1).expand(-1, self.seq_len, -1)
            out["generated_tokens"] = token_logits.argmax(dim=-1)
        return out


def build_toy_vlm_family(
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
    cfg = variants[str(variant)]
    width = max(16, int(int(cfg["width"]) * float(width_mult)))
    dim = max(int(embed_dim), int(cfg["embed"]))
    return ToyVLM(
        family=str(family),
        image_size=int(image_size),
        vocab_size=int(vocab_size),
        seq_len=int(seq_len),
        embed_dim=dim,
        width=width,
        depth=int(cfg["depth"]),
        num_classes=int(num_classes),
        dropout=float(dropout),
        architecture_mode=str(architecture_mode),
        use_instruction=bool(use_instruction),
        use_query_bridge=bool(use_query_bridge),
        use_generation_head=bool(use_generation_head),
    )


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
    out = model.forward(batch_size=2)
    shapes = {k: tuple(v.shape) for k, v in out.items() if torch.is_tensor(v)}
    print(variant, shapes)
    assert "image_embed" in out and "text_embed" in out and "logits" in out
    print("ok")


__all__ = ["ToyVLM", "build_toy_vlm_family", "smoke_test_vlm"]
