from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PromptEngineeringConfig:
    temperature: float = 0.7
    top_p: float = 1.0


@dataclass(frozen=True)
class SamplingProfile:
    temperature: float
    top_p: float


@dataclass(frozen=True)
class PromptExample:
    input_text: str
    output_text: str


@dataclass(frozen=True)
class PromptTemplate:
    instruction: str
    context: str = ""
    output_indicator: str = "Output:"

    def render(self, *, delimiter: str = "###", input_data: str = "") -> str:
        sections = [f"{delimiter} Instruction {delimiter}", str(self.instruction).strip()]
        if self.context:
            sections.extend([f"{delimiter} Context {delimiter}", str(self.context).strip()])
        if input_data:
            sections.extend([f"{delimiter} Input {delimiter}", str(input_data).strip()])
        sections.extend([f"{delimiter} Output {delimiter}", str(self.output_indicator).strip()])
        return "\n".join(sections)


class PromptEngineeringGuide:
    def __init__(self, config: PromptEngineeringConfig) -> None:
        self.config = config

    def build_few_shot_prompt(
        self,
        *,
        examples: tuple[PromptExample, ...],
        query: str,
    ) -> str:
        lines = [f"{example.input_text} // {example.output_text}" for example in examples]
        lines.append(f"{str(query).strip()} //")
        return "\n".join(lines)

    def recommend_sampling(self, *, task_type: str) -> SamplingProfile:
        task = str(task_type).strip().lower()
        if task in {"factual_qa", "classification", "extraction"}:
            return SamplingProfile(temperature=0.2, top_p=0.3)
        if task in {"creative_writing", "poetry", "brainstorming"}:
            return SamplingProfile(temperature=0.9, top_p=1.0)
        return SamplingProfile(
            temperature=float(self.config.temperature),
            top_p=float(self.config.top_p),
        )


__all__ = [
    "PromptEngineeringConfig",
    "PromptEngineeringGuide",
    "PromptExample",
    "PromptTemplate",
    "SamplingProfile",
]
