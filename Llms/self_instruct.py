from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .flan_t5 import FlanT5Config, FlanT5Model


def format_self_instruct_prompt(instruction: str, input_text: str = "") -> str:
    inst = str(instruction).strip()
    text = str(input_text).strip()
    if not text:
        return inst
    return f"{inst}\nInput: {text}"


@dataclass(frozen=True)
class SelfInstructExample:
    instruction: str
    instance_input: str
    output: str
    task_type: str
    prompt: str


@dataclass(frozen=True)
class SelfInstructInstancePrompt:
    task_type: str
    approach: str
    prompt: str


class SelfInstructDatasetBuilder:
    def __init__(self, seed_instructions: list[str], *, similarity_threshold: float = 0.7) -> None:
        self.seed_instructions = [str(item).strip() for item in seed_instructions if str(item).strip()]
        self.similarity_threshold = float(similarity_threshold)
        self._unsupported_modal_markers = (
            "image",
            "images",
            "video",
            "videos",
            "audio",
            "speech",
            "diagram",
            "draw",
            "picture",
        )

    def infer_task_type(self, instruction: str) -> str:
        lowered = str(instruction).strip().lower()
        classification_markers = ("classify", "label", "categorize", "sentiment", "choose")
        if any(marker in lowered for marker in classification_markers):
            return "classification"
        return "generation"

    def _tokenize_for_similarity(self, text: str) -> list[str]:
        return [token for token in str(text).strip().lower().replace(".", " ").split() if token]

    def rouge_l_similarity(self, first: str, second: str) -> float:
        first_tokens = self._tokenize_for_similarity(first)
        second_tokens = self._tokenize_for_similarity(second)
        if not first_tokens or not second_tokens:
            return 0.0

        dp = [[0] * (len(second_tokens) + 1) for _ in range(len(first_tokens) + 1)]
        for i, left in enumerate(first_tokens, start=1):
            for j, right in enumerate(second_tokens, start=1):
                if left == right:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        lcs = dp[-1][-1]
        precision = lcs / len(first_tokens)
        recall = lcs / len(second_tokens)
        if precision + recall == 0.0:
            return 0.0
        return 2.0 * precision * recall / (precision + recall)

    def _is_too_similar(self, candidate: str, existing: list[str]) -> bool:
        return any(
            self.rouge_l_similarity(candidate, item) >= self.similarity_threshold
            for item in existing
        )

    def is_supported_instruction(self, instruction: str) -> bool:
        lowered = str(instruction).strip().lower()
        return not any(marker in lowered for marker in self._unsupported_modal_markers)

    def filter_candidate_instructions(self, candidates: list[str]) -> list[str]:
        accepted: list[str] = []
        seen = list(self.seed_instructions)
        for raw in candidates:
            candidate = str(raw).strip()
            if not candidate:
                continue
            if not self.is_supported_instruction(candidate):
                continue
            if self._is_too_similar(candidate, seen) or self._is_too_similar(candidate, accepted):
                continue
            accepted.append(candidate)
        return accepted

    def sample_bootstrap_batch(
        self,
        *,
        machine_generated_instructions: list[str],
        seed_count: int = 6,
        generated_count: int = 2,
    ) -> tuple[str, ...]:
        selected_seed = tuple(self.seed_instructions[: int(seed_count)])
        filtered_generated = self.filter_candidate_instructions(machine_generated_instructions)
        selected_generated = tuple(filtered_generated[: int(generated_count)])
        if len(selected_seed) != int(seed_count) or len(selected_generated) != int(generated_count):
            raise ValueError("insufficient instructions to build the requested bootstrap batch")
        return selected_seed + selected_generated

    def build_bootstrap_prompt(self, instruction_batch: list[str]) -> str:
        numbered = "\n".join(
            f"{idx + 1}. {text.strip()}" for idx, text in enumerate(instruction_batch) if text.strip()
        )
        return (
            "Come up with a series of new tasks in the same spirit as the examples below.\n"
            f"{numbered}"
        ).strip()

    def build_instance_generation_prompt(
        self,
        instruction: str,
        *,
        task_type: str | None = None,
        class_labels: tuple[str, ...] = (),
    ) -> SelfInstructInstancePrompt:
        resolved_task_type = (
            self.infer_task_type(instruction) if task_type is None else str(task_type).strip().lower()
        )
        instruction_text = str(instruction).strip()
        if resolved_task_type == "classification":
            labels = tuple(str(label).strip() for label in class_labels if str(label).strip())
            label_text = ", ".join(labels) if labels else "<label>"
            prompt = (
                "Generate one classification instance for the task below.\n"
                f"Task instruction: {instruction_text}\n"
                f"Choose a class label first from: {label_text}\n"
                "Class label:\n"
                "Input:"
            )
            return SelfInstructInstancePrompt(
                task_type="classification",
                approach="output_first",
                prompt=prompt,
            )

        prompt = (
            "Generate one non-classification instance for the task below.\n"
            f"Task instruction: {instruction_text}\n"
            "Write the input first, then write the output.\n"
            "Input:\n"
            "Output:"
        )
        return SelfInstructInstancePrompt(
            task_type="generation",
            approach="input_first",
            prompt=prompt,
        )

    def build_example(
        self,
        *,
        instruction: str,
        instance_input: str,
        output: str,
        task_type: str | None = None,
    ) -> SelfInstructExample:
        resolved_task_type = self.infer_task_type(instruction) if task_type is None else str(task_type)
        return SelfInstructExample(
            instruction=str(instruction).strip(),
            instance_input=str(instance_input).strip(),
            output=str(output).strip(),
            task_type=resolved_task_type,
            prompt=format_self_instruct_prompt(instruction, instance_input),
        )


@dataclass(frozen=True)
class SelfInstructConfig:
    vocab_size: int
    max_seq_len: int
    d_model: int = 512
    num_heads: int = 8
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    d_ff: int = 2048
    dropout: float = 0.0
    similarity_threshold: float = 0.8

    def to_flan_config(self) -> FlanT5Config:
        return FlanT5Config(
            vocab_size=int(self.vocab_size),
            max_seq_len=int(self.max_seq_len),
            d_model=int(self.d_model),
            num_heads=int(self.num_heads),
            num_encoder_layers=int(self.num_encoder_layers),
            num_decoder_layers=int(self.num_decoder_layers),
            d_ff=int(self.d_ff),
            dropout=float(self.dropout),
        )


class SelfInstructModel(nn.Module):
    def __init__(
        self,
        config: SelfInstructConfig,
        *,
        seed_instructions: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.dataset_builder = SelfInstructDatasetBuilder(
            seed_instructions or [],
            similarity_threshold=float(config.similarity_threshold),
        )
        self.base_model = FlanT5Model(config.to_flan_config())

    def build_training_prompt(self, instruction: str, input_text: str = "") -> str:
        return format_self_instruct_prompt(instruction, input_text)

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        decoder_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.base_model(
            input_ids=input_ids,
            decoder_input_ids=decoder_input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
        )


__all__ = [
    "SelfInstructConfig",
    "SelfInstructDatasetBuilder",
    "SelfInstructExample",
    "SelfInstructInstancePrompt",
    "SelfInstructModel",
    "format_self_instruct_prompt",
]
