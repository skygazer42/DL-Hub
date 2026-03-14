from __future__ import annotations

import re
from dataclasses import dataclass


PERMISSIVE_LICENSES = frozenset(
    {
        "Apache-2.0",
        "BSD-2-Clause",
        "BSD-3-Clause",
        "CC0-1.0",
        "ISC",
        "MIT",
        "Unlicense",
    }
)


@dataclass(frozen=True)
class TheStackConfig:
    permissive_only: bool = True
    near_dedup_threshold: float = 0.85


@dataclass(frozen=True)
class StackFile:
    repo_name: str
    path: str
    language: str
    license: str
    content: str


class NearDeduplicator:
    def __init__(self, threshold: float) -> None:
        if not 0.0 <= float(threshold) <= 1.0:
            raise ValueError("threshold must be between 0 and 1")
        self.threshold = float(threshold)

    @staticmethod
    def token_set(content: str) -> set[str]:
        return set(re.findall(r"[A-Za-z0-9_]+", content.lower()))

    def similarity(self, left: StackFile, right: StackFile) -> float:
        left_tokens = self.token_set(left.content)
        right_tokens = self.token_set(right.content)
        if not left_tokens and not right_tokens:
            return 1.0
        if not left_tokens or not right_tokens:
            return 0.0
        intersection = len(left_tokens & right_tokens)
        union = len(left_tokens | right_tokens)
        return intersection / union

    def deduplicate(self, files: tuple[StackFile, ...]) -> tuple[StackFile, ...]:
        kept: list[StackFile] = []
        for candidate in files:
            if any(self.similarity(candidate, existing) >= self.threshold for existing in kept):
                continue
            kept.append(candidate)
        return tuple(kept)


class TheStackDataset:
    def __init__(
        self,
        files: list[StackFile] | tuple[StackFile, ...],
        config: TheStackConfig | None = None,
    ) -> None:
        self.files = tuple(files)
        self.config = config or TheStackConfig()

    def permissive_subset(self) -> "TheStackDataset":
        filtered = [
            file
            for file in self.files
            if file.license in PERMISSIVE_LICENSES
        ]
        return TheStackDataset(filtered, self.config)

    def near_deduplicate(self) -> "TheStackDataset":
        deduplicator = NearDeduplicator(self.config.near_dedup_threshold)
        return TheStackDataset(deduplicator.deduplicate(self.files), self.config)

    def remove_repositories(self, repo_names: set[str] | tuple[str, ...] | list[str]) -> "TheStackDataset":
        excluded = set(repo_names)
        kept = [file for file in self.files if file.repo_name not in excluded]
        return TheStackDataset(kept, self.config)

    def language_bytes(self) -> dict[str, int]:
        totals: dict[str, int] = {}
        for file in self.files:
            totals.setdefault(file.language, 0)
            totals[file.language] += len(file.content.encode("utf-8"))
        return totals


__all__ = [
    "NearDeduplicator",
    "PERMISSIVE_LICENSES",
    "StackFile",
    "TheStackConfig",
    "TheStackDataset",
]
