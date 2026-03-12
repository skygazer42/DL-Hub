from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SurveyAspect:
    name: str
    guiding_question: str


@dataclass(frozen=True)
class BenchmarkSuite:
    name: str
    focus: str


@dataclass(frozen=True)
class SurveyResource:
    kind: str
    examples: tuple[str, ...]


class LLMSurveyGuide:
    def major_aspects(self) -> tuple[SurveyAspect, ...]:
        return (
            SurveyAspect("pre-training", "how to pre-train a capable LLM"),
            SurveyAspect("adaptation tuning", "how to adapt LLMs for effectiveness and safety"),
            SurveyAspect("utilization", "how to use LLMs effectively"),
            SurveyAspect("capacity evaluation", "how to evaluate LLM abilities holistically"),
        )

    def benchmarks(self) -> tuple[BenchmarkSuite, ...]:
        return (
            BenchmarkSuite("MMLU", "multi-task knowledge understanding"),
            BenchmarkSuite("BIG-bench", "broad challenging task evaluation"),
            BenchmarkSuite("HELM", "holistic multi-metric evaluation"),
        )

    def resources(self) -> tuple[SurveyResource, ...]:
        return (
            SurveyResource("checkpoints", ("LLaMA", "Flan-T5", "BLOOM")),
            SurveyResource("corpora", ("C4", "The Pile", "ROOTS")),
            SurveyResource("tooling", ("DeepSpeed", "BMTrain", "Evaluation Harness")),
        )


__all__ = [
    "BenchmarkSuite",
    "LLMSurveyGuide",
    "SurveyAspect",
    "SurveyResource",
]
