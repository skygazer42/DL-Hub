from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RedPajamaSlice:
    name: str
    relative_weight: float
    license_filtered: bool = False
    quality_pipeline: str = "standard"


@dataclass(frozen=True)
class RedPajamaConfig:
    total_tokens: int = 1_200_000_000_000
    target_dataset: str = "llama-reproduction"
    uses_ccnet: bool = True


def canonical_red_pajama_slices() -> tuple[RedPajamaSlice, ...]:
    return (
        RedPajamaSlice("CommonCrawl", 0.42, quality_pipeline="ccnet+quality-filters"),
        RedPajamaSlice("C4", 0.13),
        RedPajamaSlice("GitHub", 0.12, license_filtered=True, quality_pipeline="license+quality"),
        RedPajamaSlice("arXiv", 0.08, quality_pipeline="boilerplate-removal"),
        RedPajamaSlice("Books", 0.09, quality_pipeline="deduplication"),
        RedPajamaSlice("Wikipedia", 0.08, quality_pipeline="boilerplate-removal"),
        RedPajamaSlice("StackExchange", 0.08, quality_pipeline="boilerplate-removal"),
    )


class RedPajamaDataset:
    def __init__(
        self,
        config: RedPajamaConfig,
        slices: tuple[RedPajamaSlice, ...] | list[RedPajamaSlice],
    ) -> None:
        if not slices:
            raise ValueError("slices cannot be empty")
        self.config = config
        self.slices = tuple(slices)

    def license_filtered_sources(self) -> tuple[str, ...]:
        return tuple(data_slice.name for data_slice in self.slices if data_slice.license_filtered)

    def allocate_tokens(self, total_units: int) -> dict[str, int]:
        if int(total_units) < 0:
            raise ValueError("total_units must be >= 0")
        raw = {
            data_slice.name: float(data_slice.relative_weight) * int(total_units)
            for data_slice in self.slices
        }
        counts = {name: int(value) for name, value in raw.items()}
        remaining = int(total_units) - sum(counts.values())
        remainders = sorted(
            ((raw[name] - counts[name], name) for name in counts),
            key=lambda item: (-item[0], item[1]),
        )
        for _, name in remainders[:remaining]:
            counts[name] += 1
        return counts


__all__ = [
    "RedPajamaConfig",
    "RedPajamaDataset",
    "RedPajamaSlice",
    "canonical_red_pajama_slices",
]
