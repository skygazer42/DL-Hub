"""VLM timeline metadata (best effort, for docs and CLI)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TimelineEntry:
    year: int | None
    family: str
    method: str
    group: str
    reference: str | None = None


_ENTRIES: list[TimelineEntry] = [
    TimelineEntry(
        2021, "vilt", "ViLT (single-stream vision-language transformer)", "single_stream"
    ),
    TimelineEntry(2021, "clip", "CLIP (contrastive image-text pretraining)", "dual_encoder"),
    TimelineEntry(2021, "align", "ALIGN (large-scale dual-encoder alignment)", "dual_encoder"),
    TimelineEntry(2021, "albef", "ALBEF (align before fuse)", "fusion_encoder_decoder"),
    TimelineEntry(
        2021,
        "simvlm",
        "SimVLM (simple visual language model with prefix LM)",
        "fusion_encoder_decoder",
    ),
    TimelineEntry(2021, "lit", "LiT (locked-image text tuning)", "dual_encoder"),
    TimelineEntry(
        2022, "ofa", "OFA (unified sequence-to-sequence multimodal model)", "fusion_encoder_decoder"
    ),
    TimelineEntry(
        2022, "blip", "BLIP (bootstrapped language-image pretraining)", "fusion_encoder_decoder"
    ),
    TimelineEntry(2022, "coca", "CoCa (contrastive captioner)", "fusion_encoder_decoder"),
    TimelineEntry(2022, "flamingo", "Flamingo (few-shot visual language model)", "multimodal_llm"),
    TimelineEntry(2022, "pali", "PaLI (scaling language-image learning)", "fusion_encoder_decoder"),
    TimelineEntry(2023, "blip2", "BLIP-2 (querying transformer bridge)", "multimodal_llm"),
    TimelineEntry(
        2023, "instructblip", "InstructBLIP (instruction-aware BLIP-2)", "multimodal_llm"
    ),
    TimelineEntry(2023, "llava", "LLaVA (visual instruction tuning)", "multimodal_llm"),
    TimelineEntry(2023, "kosmos2", "Kosmos-2 (grounded multimodal LLM)", "multimodal_llm"),
    TimelineEntry(
        2023, "pali_x", "PaLI-X (multilingual transfer at scale)", "fusion_encoder_decoder"
    ),
    TimelineEntry(2023, "minigpt4", "MiniGPT-4 (visual instruction alignment)", "multimodal_llm"),
    TimelineEntry(
        2023, "mplug_owl2", "mPLUG-Owl2 (modular multimodal chat model)", "multimodal_llm"
    ),
    TimelineEntry(2023, "qwen_vl", "Qwen-VL (multilingual multimodal assistant)", "multimodal_llm"),
    TimelineEntry(2023, "cogvlm", "CogVLM (visual expert large language model)", "multimodal_llm"),
]


def entries() -> list[TimelineEntry]:
    return list(_ENTRIES)


def by_family() -> dict[str, TimelineEntry]:
    return {entry.family: entry for entry in _ENTRIES}
