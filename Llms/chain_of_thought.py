from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True)
class ChainOfThoughtExample:
    question: str
    rationale: str = ""
    answer: str = ""


@dataclass(frozen=True)
class ChainOfThoughtConfig:
    temperature: float = 0.4
    num_samples: int = 5
    reasoning_cue: str = "Let's think step by step."


def format_chain_of_thought_prompt(
    *,
    question: str,
    demonstrations: tuple[ChainOfThoughtExample, ...] = (),
    reasoning_cue: str = "Let's think step by step.",
) -> str:
    sections: list[str] = []
    for demo in demonstrations:
        parts = [f"Question: {demo.question}"]
        if demo.rationale:
            parts.append(f"Reasoning: {demo.rationale}")
        if demo.answer:
            parts.append(f"Answer: {demo.answer}")
        sections.append("\n".join(parts))

    sections.append(f"Question: {question}\nReasoning: {reasoning_cue}")
    return "\n\n".join(sections).strip()


def extract_final_answer(reasoning_trace: str) -> str:
    trace = str(reasoning_trace).strip()
    if not trace:
        return ""

    patterns = (
        r"answer\s*:\s*([^\n]+)",
        r"the answer is\s+([^\n\.]+)",
    )
    matches: list[str] = []
    for pattern in patterns:
        matches.extend(
            match.strip(" .,:;")
            for match in re.findall(pattern, trace, flags=re.IGNORECASE)
            if match.strip()
        )
    if matches:
        return matches[-1]

    lines = [line.strip() for line in trace.splitlines() if line.strip()]
    if lines:
        return lines[-1].strip(" .,:;")
    return trace


class SelfConsistencyDecoder:
    strategy = "sample+vote"

    def __init__(self, temperature: float = 0.4, num_samples: int = 5) -> None:
        self.temperature = float(temperature)
        self.num_samples = int(num_samples)

    def majority_vote(self, reasoning_traces: tuple[str, ...] | list[str]) -> str:
        if not reasoning_traces:
            raise ValueError("reasoning_traces must not be empty")

        answers = [extract_final_answer(trace) for trace in reasoning_traces]
        counts = Counter(answer for answer in answers if answer)
        if not counts:
            return ""

        best_count = max(counts.values())
        for answer in answers:
            if counts.get(answer, 0) == best_count:
                return answer
        return answers[0]


class ChainOfThoughtReasoner:
    prompting_method = "few-shot-chain-of-thought"

    def __init__(self, config: ChainOfThoughtConfig) -> None:
        self.config = config
        self.decoder = SelfConsistencyDecoder(
            temperature=float(config.temperature),
            num_samples=int(config.num_samples),
        )

    def build_prompt(
        self,
        *,
        question: str,
        demonstrations: tuple[ChainOfThoughtExample, ...] = (),
    ) -> str:
        return format_chain_of_thought_prompt(
            question=question,
            demonstrations=demonstrations,
            reasoning_cue=self.config.reasoning_cue,
        )

    def aggregate_answers(self, reasoning_traces: tuple[str, ...] | list[str]) -> str:
        return self.decoder.majority_vote(reasoning_traces)


__all__ = [
    "ChainOfThoughtConfig",
    "ChainOfThoughtExample",
    "ChainOfThoughtReasoner",
    "SelfConsistencyDecoder",
    "extract_final_answer",
    "format_chain_of_thought_prompt",
]
