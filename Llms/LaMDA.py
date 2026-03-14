from __future__ import annotations

import re
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from ._shared import causal_mask, expand_key_padding_mask, make_attention_mask
from .t5 import T5RelativePositionBias


class GatedGELUMLP(nn.Module):
    activation_name = "gated_gelu"

    def __init__(self, hidden_size: int, intermediate_size: int, *, dropout: float = 0.0) -> None:
        super().__init__()
        self.wi_0 = nn.Linear(int(hidden_size), int(intermediate_size), bias=False)
        self.wi_1 = nn.Linear(int(hidden_size), int(intermediate_size), bias=False)
        self.wo = nn.Linear(int(intermediate_size), int(hidden_size), bias=False)
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gated = F.gelu(self.wi_0(x)) * self.wi_1(x)
        return self.wo(self.dropout(gated))


@dataclass(frozen=True)
class LaMDAConfig:
    vocab_size: int
    max_seq_len: int
    hidden_size: int = 512
    num_heads: int = 8
    num_layers: int = 6
    intermediate_size: int | None = None
    dropout: float = 0.0
    layer_norm_eps: float = 1e-6
    relative_attention_num_buckets: int = 32
    relative_attention_max_distance: int = 128


@dataclass(frozen=True)
class LaMDAQualityScore:
    sensibleness: float
    specificity: float
    interestingness: float

    @property
    def ssi(self) -> float:
        return (
            float(self.sensibleness)
            + float(self.specificity)
            + float(self.interestingness)
        ) / 3.0


@dataclass(frozen=True)
class LaMDAResponse:
    text: str
    safety: float
    groundedness: float
    quality: LaMDAQualityScore
    citations: tuple[str, ...] = ()


class LaMDAAttention(nn.Module):
    def __init__(self, config: LaMDAConfig) -> None:
        super().__init__()
        if int(config.hidden_size) % int(config.num_heads) != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.hidden_size = int(config.hidden_size)
        self.num_heads = int(config.num_heads)
        self.head_dim = int(config.hidden_size // config.num_heads)
        self.relative_attention_bias = T5RelativePositionBias(
            self.num_heads,
            num_buckets=int(config.relative_attention_num_buckets),
            max_distance=int(config.relative_attention_max_distance),
        )
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.dropout = nn.Dropout(float(config.dropout))

    def _shape(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        return x.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        q = self._shape(self.q_proj(x))
        k = self._shape(self.k_proj(x))
        v = self._shape(self.v_proj(x))

        scores = torch.matmul(q, k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        scores = scores + self.relative_attention_bias(seq_len, seq_len, device=x.device)
        scores = scores.masked_fill(
            ~expand_key_padding_mask(attention_mask, batch_size=bsz, seq_len=seq_len),
            torch.finfo(scores.dtype).min,
        )
        scores = scores.masked_fill(
            ~causal_mask(seq_len, device=x.device),
            torch.finfo(scores.dtype).min,
        )
        probs = torch.softmax(scores, dim=-1)
        probs = self.dropout(probs)
        out = torch.matmul(probs, v)
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.hidden_size)
        out = self.o_proj(out)
        return out * attention_mask.unsqueeze(-1).to(dtype=out.dtype)


class LaMDABlock(nn.Module):
    def __init__(self, config: LaMDAConfig) -> None:
        super().__init__()
        hidden_size = int(config.hidden_size)
        intermediate = (
            int(config.intermediate_size)
            if config.intermediate_size is not None
            else hidden_size * 4
        )
        self.self_attention_layer_norm = nn.LayerNorm(
            hidden_size, eps=float(config.layer_norm_eps)
        )
        self.self_attention = LaMDAAttention(config)
        self.ff_layer_norm = nn.LayerNorm(hidden_size, eps=float(config.layer_norm_eps))
        self.feed_forward = GatedGELUMLP(
            hidden_size,
            intermediate,
            dropout=float(config.dropout),
        )
        self.dropout = nn.Dropout(float(config.dropout))

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.self_attention(self.self_attention_layer_norm(x), attention_mask))
        x = x + self.dropout(self.feed_forward(self.ff_layer_norm(x)))
        return x


class LaMDAToolset:
    def __init__(self) -> None:
        self.tool_names = ["calculator", "translator", "retrieval"]
        self._translations = {"hello in french": "Bonjour"}
        self._retrievals = {"how old is rafael nadal?": "Rafael Nadal / Age / 35"}

    def route_query(self, query: str) -> str:
        stripped = query.strip()
        if re.fullmatch(r"[0-9+\-*/(). ]+", stripped):
            return "calculator"
        lowered = stripped.lower()
        if " in french" in lowered:
            return "translator"
        return "retrieval"

    def run(self, tool_name: str, query: str) -> list[str]:
        if tool_name == "calculator":
            if not re.fullmatch(r"[0-9+\-*/(). ]+", query.strip()):
                return []
            result = eval(query, {"__builtins__": {}}, {})
            if isinstance(result, float) and result.is_integer():
                result = int(result)
            return [str(result)]
        if tool_name == "translator":
            translated = self._translations.get(query.strip().lower())
            return [] if translated is None else [translated]
        if tool_name == "retrieval":
            retrieved = self._retrievals.get(query.strip().lower())
            return [] if retrieved is None else [retrieved]
        raise ValueError(f"unknown tool: {tool_name}")


class LaMDAModel(nn.Module):
    default_tool_order = ("calculator", "translator", "retrieval")

    def __init__(self, config: LaMDAConfig, *, toolset: LaMDAToolset | None = None) -> None:
        super().__init__()
        self.config = config
        self.toolset = LaMDAToolset() if toolset is None else toolset
        self.embed_tokens = nn.Embedding(int(config.vocab_size), int(config.hidden_size))
        self.layers = nn.ModuleList([LaMDABlock(config) for _ in range(int(config.num_layers))])
        self.final_layer_norm = nn.LayerNorm(
            int(config.hidden_size), eps=float(config.layer_norm_eps)
        )
        self.lm_head = nn.Linear(int(config.hidden_size), int(config.vocab_size), bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        mask = make_attention_mask(input_ids, attention_mask)
        x = self.embed_tokens(input_ids.to(torch.long))
        for layer in self.layers:
            x = layer(x, mask)
        x = self.final_layer_norm(x)
        return self.lm_head(x)


class LaMDADialogAgent:
    def __init__(self, *, toolset: LaMDAToolset | None = None) -> None:
        self.toolset = LaMDAToolset() if toolset is None else toolset
        self._generic_responses = {
            "i am not sure",
            "i'm not sure",
            "me too",
            "ok",
            "okay",
        }
        self._unsafe_patterns = (
            "illegal drugs",
            "kill yourself",
            "hate all",
            "buy weapons",
        )
        self._stopwords = {
            "a",
            "an",
            "and",
            "are",
            "do",
            "i",
            "in",
            "is",
            "it",
            "love",
            "me",
            "my",
            "of",
            "the",
            "to",
            "too",
            "user",
            "you",
        }

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", str(text).lower())

    def score_quality(self, context: str, response: str) -> LaMDAQualityScore:
        text = str(response).strip()
        lowered = text.lower()
        sensible = 1.0 if text else 0.0

        context_tokens = {
            token for token in self._tokenize(context) if token not in self._stopwords
        }
        response_tokens = {
            token for token in self._tokenize(response) if token not in self._stopwords
        }
        specificity = 1.0 if context_tokens & response_tokens else 0.0

        generic_key = lowered.rstrip(".!?")
        interestingness = 1.0 if len(response_tokens) >= 4 and generic_key not in self._generic_responses else 0.0
        return LaMDAQualityScore(
            sensibleness=sensible,
            specificity=specificity,
            interestingness=interestingness,
        )

    def score_safety(self, response: str) -> float:
        lowered = str(response).lower()
        return 0.0 if any(pattern in lowered for pattern in self._unsafe_patterns) else 1.0

    def build_grounded_response(self, query: str) -> LaMDAResponse | None:
        tool_name = self.toolset.route_query(query)
        outputs = self.toolset.run(tool_name, query)
        if not outputs:
            return None
        value = outputs[0]
        if tool_name == "calculator":
            text = f"That would be {value}. [{tool_name}]"
        elif tool_name == "translator":
            text = f"In French, that is {value}. [{tool_name}]"
        else:
            text = f"According to retrieval, {value}. [{tool_name}]"
        quality = self.score_quality(query, text)
        return LaMDAResponse(
            text=text,
            safety=1.0,
            groundedness=1.0,
            quality=quality,
            citations=(f"{tool_name}://result",),
        )

    def respond(
        self,
        query: str,
        *,
        candidate_responses: list[str] | tuple[str, ...],
    ) -> LaMDAResponse:
        scored: list[LaMDAResponse] = []
        for text in candidate_responses:
            quality = self.score_quality(query, text)
            scored.append(
                LaMDAResponse(
                    text=str(text),
                    safety=self.score_safety(text),
                    groundedness=0.0,
                    quality=quality,
                    citations=(),
                )
            )

        grounded = self.build_grounded_response(query)
        if grounded is not None:
            scored.append(grounded)

        safe = [item for item in scored if item.safety > 0.0]
        if not safe:
            raise ValueError("No safe candidate responses available")
        return max(safe, key=lambda item: (item.groundedness, item.quality.ssi, len(item.text)))


__all__ = [
    "GatedGELUMLP",
    "LaMDAAttention",
    "LaMDABlock",
    "LaMDAConfig",
    "LaMDADialogAgent",
    "LaMDAModel",
    "LaMDAQualityScore",
    "LaMDAResponse",
    "LaMDAToolset",
]
