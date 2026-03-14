from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AIBubbleEntry:
    label: str
    parameters_b: float
    availability: str
    chinchilla_scale: bool = False


def canonical_ai_bubbles_entries() -> tuple[AIBubbleEntry, ...]:
    return (
        AIBubbleEntry("Bard", 137.0, "closed"),
        AIBubbleEntry("GPT-4", 0.0, "closed"),
        AIBubbleEntry("LLaMA", 65.0, "open", chinchilla_scale=True),
        AIBubbleEntry("PaLM", 540.0, "closed"),
        AIBubbleEntry("Chinchilla", 70.0, "closed", chinchilla_scale=True),
        AIBubbleEntry("Flamingo", 80.0, "closed", chinchilla_scale=True),
    )


class AIBubblesRegistry:
    def __init__(self, entries: tuple[AIBubbleEntry, ...] | list[AIBubbleEntry]) -> None:
        self.entries = tuple(entries)

    def open_models(self) -> tuple[AIBubbleEntry, ...]:
        return tuple(entry for entry in self.entries if entry.availability == "open")

    def closed_models(self) -> tuple[AIBubbleEntry, ...]:
        return tuple(entry for entry in self.entries if entry.availability == "closed")

    def chinchilla_scale_models(self) -> tuple[AIBubbleEntry, ...]:
        return tuple(entry for entry in self.entries if entry.chinchilla_scale)


__all__ = [
    "AIBubbleEntry",
    "AIBubblesRegistry",
    "canonical_ai_bubbles_entries",
]
