from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TimelineEntry:
    year: int
    month: int
    label: str
    category: str


def canonical_llm_timeline_entries() -> tuple[TimelineEntry, ...]:
    return (
        TimelineEntry(2023, 5, "OpenLLaMA", "model"),
        TimelineEntry(2023, 5, "RedPajama-INCITE-Base", "model"),
        TimelineEntry(2023, 4, "Pythia", "model"),
        TimelineEntry(2023, 4, "RedPajama-Data-1T", "dataset"),
        TimelineEntry(2023, 3, "Bard", "model"),
        TimelineEntry(2023, 3, "OpenAssistant", "dataset"),
        TimelineEntry(2023, 3, "StarCoderData", "dataset"),
        TimelineEntry(2023, 3, "GPT4All", "model"),
        TimelineEntry(2023, 2, "GPT4All-J", "model"),
        TimelineEntry(2023, 2, "StarCoder", "model"),
        TimelineEntry(2023, 2, "OASST1", "dataset"),
        TimelineEntry(2023, 2, "Survey on ChatGPT and Beyond", "survey"),
        TimelineEntry(2023, 1, "BLIP-2", "model"),
    )


class LLMTimeline:
    def __init__(self, entries: tuple[TimelineEntry, ...] | list[TimelineEntry]) -> None:
        self.entries = tuple(entries)

    def filter(
        self,
        *,
        year: int | None = None,
        month: int | None = None,
    ) -> tuple[TimelineEntry, ...]:
        filtered = self.entries
        if year is not None:
            filtered = tuple(entry for entry in filtered if entry.year == int(year))
        if month is not None:
            filtered = tuple(entry for entry in filtered if entry.month == int(month))
        return filtered


__all__ = [
    "LLMTimeline",
    "TimelineEntry",
    "canonical_llm_timeline_entries",
]
