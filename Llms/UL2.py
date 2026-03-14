from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .t5 import T5Config, T5Model


@dataclass(frozen=True)
class UL2Mode:
    name: str
    tag: str
    description: str
    noise_density: float
    mean_noise_span_length: float


@dataclass(frozen=True)
class UL2NoisingPlan:
    mode: str
    tag: str
    objective_name: str
    noise_density: float
    mean_noise_span_length: float
    num_masked_tokens: int
    num_spans: int
    prefix_length: int | None = None


@dataclass(frozen=True)
class UL2NoisedSample:
    corrupted_input: list[int]
    target: list[int]
    plan: UL2NoisingPlan


class UL2Objective:
    def __init__(self) -> None:
        self.modes = {
            "R": UL2Mode(
                name="R",
                tag="[NLU]",
                description="regular denoising / short spans & low corruption",
                noise_density=0.15,
                mean_noise_span_length=3.0,
            ),
            "S": UL2Mode(
                name="S",
                tag="[S2S]",
                description="sequential denoising / prefix language modeling",
                noise_density=0.15,
                mean_noise_span_length=3.0,
            ),
            "X": UL2Mode(
                name="X",
                tag="[NLG]",
                description="extreme denoising / long spans & high corruption",
                noise_density=0.5,
                mean_noise_span_length=32.0,
            ),
        }
        self.mode_to_tag = {k: v.tag for k, v in self.modes.items()}

    def get_mode(self, mode: str) -> UL2Mode:
        key = str(mode).upper().strip()
        if key not in self.modes:
            raise ValueError(f"Unknown UL2 mode: {mode}")
        return self.modes[key]

    def format_with_mode(self, text: str, *, mode: str) -> str:
        mode_cfg = self.get_mode(mode)
        payload = str(text).strip()
        return f"{mode_cfg.tag} {payload}".strip()

    def apply_mode(
        self,
        tokens: list[int] | tuple[int, ...],
        *,
        mode: str,
        sentinel_start: int = 32000,
    ) -> UL2NoisedSample:
        values = [int(token) for token in tokens]
        if not values:
            raise ValueError("tokens must not be empty")
        mode_cfg = self.get_mode(mode)
        if mode_cfg.name == "S":
            prefix_length = max(1, len(values) // 2)
            plan = UL2NoisingPlan(
                mode=mode_cfg.name,
                tag=mode_cfg.tag,
                objective_name="prefix_lm",
                noise_density=float(mode_cfg.noise_density),
                mean_noise_span_length=float(mode_cfg.mean_noise_span_length),
                num_masked_tokens=len(values) - prefix_length,
                num_spans=1,
                prefix_length=prefix_length,
            )
            return UL2NoisedSample(
                corrupted_input=values[:prefix_length],
                target=values[prefix_length:],
                plan=plan,
            )

        objective_name = (
            "regular_span_corruption"
            if mode_cfg.name == "R"
            else "extreme_span_corruption"
        )
        return self._apply_span_corruption(
            values,
            mode_cfg=mode_cfg,
            objective_name=objective_name,
            sentinel_start=int(sentinel_start),
        )

    def _apply_span_corruption(
        self,
        tokens: list[int],
        *,
        mode_cfg: UL2Mode,
        objective_name: str,
        sentinel_start: int,
    ) -> UL2NoisedSample:
        seq_len = len(tokens)
        num_masked_tokens = max(1, int(round(seq_len * float(mode_cfg.noise_density))))
        num_masked_tokens = min(num_masked_tokens, seq_len - 1 if seq_len > 1 else 1)
        mean_span = max(1, int(round(float(mode_cfg.mean_noise_span_length))))
        num_spans = max(1, int(round(num_masked_tokens / mean_span)))
        spans = self._deterministic_spans(
            seq_len,
            num_masked_tokens=num_masked_tokens,
            num_spans=num_spans,
        )

        corrupted_input: list[int] = []
        target: list[int] = []
        cursor = 0
        for idx, (start, end) in enumerate(spans):
            corrupted_input.extend(tokens[cursor:start])
            sentinel = int(sentinel_start) + idx
            corrupted_input.append(sentinel)
            target.append(sentinel)
            target.extend(tokens[start:end])
            cursor = end
        corrupted_input.extend(tokens[cursor:])

        plan = UL2NoisingPlan(
            mode=mode_cfg.name,
            tag=mode_cfg.tag,
            objective_name=str(objective_name),
            noise_density=float(mode_cfg.noise_density),
            mean_noise_span_length=float(mode_cfg.mean_noise_span_length),
            num_masked_tokens=num_masked_tokens,
            num_spans=len(spans),
        )
        return UL2NoisedSample(
            corrupted_input=corrupted_input,
            target=target,
            plan=plan,
        )

    def _deterministic_spans(
        self,
        seq_len: int,
        *,
        num_masked_tokens: int,
        num_spans: int,
    ) -> list[tuple[int, int]]:
        remaining_tokens = int(num_masked_tokens)
        remaining_spans = int(num_spans)
        starts = [int(i * seq_len / max(1, num_spans)) for i in range(num_spans)]
        spans: list[tuple[int, int]] = []
        used_until = 0
        for start_hint in starts:
            span_len = max(1, remaining_tokens // remaining_spans)
            start = max(int(start_hint), used_until)
            max_end = seq_len - (remaining_spans - 1)
            end = min(start + span_len, max_end)
            if end <= start:
                end = min(seq_len, start + 1)
            spans.append((start, end))
            used_until = end
            remaining_tokens -= end - start
            remaining_spans -= 1
        if remaining_tokens > 0 and spans:
            start, end = spans[-1]
            spans[-1] = (start, min(seq_len, end + remaining_tokens))
        return spans


@dataclass(frozen=True)
class UL2Config:
    vocab_size: int
    max_seq_len: int
    d_model: int = 512
    num_heads: int = 8
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    d_ff: int = 2048
    dropout: float = 0.0

    def to_t5_config(self) -> T5Config:
        return T5Config(
            vocab_size=int(self.vocab_size),
            max_seq_len=int(self.max_seq_len),
            d_model=int(self.d_model),
            num_heads=int(self.num_heads),
            num_encoder_layers=int(self.num_encoder_layers),
            num_decoder_layers=int(self.num_decoder_layers),
            d_ff=int(self.d_ff),
            dropout=float(self.dropout),
        )


class UL2Model(nn.Module):
    def __init__(self, config: UL2Config) -> None:
        super().__init__()
        self.config = config
        self.objective = UL2Objective()
        self.base_model = T5Model(config.to_t5_config())

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        mode: str = "R",
        attention_mask: torch.Tensor | None = None,
        decoder_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self.objective.get_mode(mode)
        return self.base_model(
            input_ids=input_ids,
            decoder_input_ids=decoder_input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
        )


__all__ = [
    "UL2Config",
    "UL2Mode",
    "UL2Model",
    "UL2NoisedSample",
    "UL2NoisingPlan",
    "UL2Objective",
]
