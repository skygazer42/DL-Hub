from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PileComponent:
    name: str
    effective_size_gib: float


@dataclass(frozen=True)
class PileConfig:
    total_size_gib: float = 825.18
    deduplicated: bool = True


def canonical_pile_components() -> tuple[PileComponent, ...]:
    return (
        PileComponent("Pile-CC", 227.12),
        PileComponent("PubMed Central", 90.27),
        PileComponent("Books3", 100.96),
        PileComponent("OpenWebText2", 62.77),
        PileComponent("ArXiv", 56.21),
        PileComponent("Github", 95.16),
        PileComponent("FreeLaw", 51.15),
        PileComponent("Stack Exchange", 32.20),
        PileComponent("USPTO Backgrounds", 22.90),
        PileComponent("PubMed Abstracts", 19.26),
        PileComponent("Gutenberg (PG-19)", 10.88),
        PileComponent("OpenSubtitles", 12.98),
        PileComponent("Wikipedia (en)", 6.38),
        PileComponent("DM Mathematics", 7.75),
        PileComponent("Ubuntu IRC", 5.52),
        PileComponent("BookCorpus2", 6.30),
        PileComponent("EuroParl", 4.59),
        PileComponent("HackerNews", 3.90),
        PileComponent("YoutubeSubtitles", 3.73),
        PileComponent("PhilPapers", 2.38),
        PileComponent("NIH ExPorter", 1.89),
        PileComponent("Enron Emails", 0.88),
    )


class PileMixture:
    def __init__(
        self,
        config: PileConfig,
        components: tuple[PileComponent, ...] | list[PileComponent],
    ) -> None:
        if not components:
            raise ValueError("components cannot be empty")
        self.config = config
        self.components = tuple(components)

    def normalized_shares(self) -> dict[str, float]:
        total = sum(component.effective_size_gib for component in self.components)
        if total <= 0.0:
            raise ValueError("sum of component sizes must be > 0")
        return {
            component.name: component.effective_size_gib / total
            for component in self.components
        }

    def allocate(self, total_units: int) -> dict[str, int]:
        if int(total_units) < 0:
            raise ValueError("total_units must be >= 0")
        shares = self.normalized_shares()
        raw_counts = {
            name: shares[name] * int(total_units)
            for name in shares
        }
        counts = {name: int(raw) for name, raw in raw_counts.items()}
        remaining = int(total_units) - sum(counts.values())
        remainders = sorted(
            ((raw_counts[name] - counts[name], name) for name in counts),
            key=lambda item: (-item[0], item[1]),
        )
        for _, name in remainders[:remaining]:
            counts[name] += 1
        return counts


__all__ = [
    "PileComponent",
    "PileConfig",
    "PileMixture",
    "canonical_pile_components",
]
