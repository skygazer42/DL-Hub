from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LLMResourceEntry:
    filename: str
    title: str
    module_names: tuple[str, ...]
    status: str = "implemented"
    notes: str = ""


_ENTRIES: tuple[LLMResourceEntry, ...] = (
    LLMResourceEntry(
        "2023-Alan-D-Thompson-AI-Bubbles-Rev-7.pdf",
        "AI Bubbles",
        ("ai_bubbles",),
        notes="Market and capability reference mapped to the AI bubbles registry helper.",
    ),
    LLMResourceEntry(
        "2023_GPT4All_Technical_Report.pdf",
        "GPT4All Technical Report",
        ("gpt4all",),
    ),
    LLMResourceEntry("Anthropic.pdf", "Anthropic HHH / Constitutional-style model notes", ("anthropic",)),
    LLMResourceEntry("Blip2.pdf", "BLIP-2", ("blip2",)),
    LLMResourceEntry("Bloom.pdf", "BLOOM", ("bloom",)),
    LLMResourceEntry(
        "Chinchilia .pdf",
        "Chinchilla",
        ("chinchilla",),
        notes="Resource filename uses a typo; implementation uses the canonical paper spelling.",
    ),
    LLMResourceEntry("Dolly_.pdf", "Dolly", ("dolly",)),
    LLMResourceEntry("FED.pdf", "Fast Ensemble Decoding", ("fed",)),
    LLMResourceEntry("Flamingo.pdf", "Flamingo", ("flamingo",)),
    LLMResourceEntry("Flan- T5.pdf", "Flan-T5", ("flan_t5",)),
    LLMResourceEntry("GPT4All-J.pdf", "GPT4All-J", ("gpt4all_j",)),
    LLMResourceEntry("Helm.pdf", "HELM", ("helm",)),
    LLMResourceEntry("Imagen.pdf", "Imagen", ("imagen",)),
    LLMResourceEntry("Instructgpt.pdf", "InstructGPT", ("instructgpt",)),
    LLMResourceEntry("LLM surveys .pdf", "LLM Survey", ("llm_survey",)),
    LLMResourceEntry("LLaMA base model.pdf", "LLaMA Base Model", ("llama",)),
    LLMResourceEntry("LLaMA-Adapter.pdf", "LLaMA-Adapter", ("llama_adapter",)),
    LLMResourceEntry("LLaMA.pdf", "LLaMA", ("llama",)),
    LLMResourceEntry("Lamda.pdf", "LaMDA", ("lamda",)),
    LLMResourceEntry("Lora.pdf", "LoRA", ("lora",)),
    LLMResourceEntry("MTF.pdf", "Multitask Prompted Training", ("mtf",)),
    LLMResourceEntry("Megatron.pdf", "Megatron-LM", ("megatron",)),
    LLMResourceEntry(
        "PaLM (Scaling Language Modeling with Pathways).md",
        "PaLM",
        ("palm",),
        notes="Markdown summary alongside the PaLM paper PDF.",
    ),
    LLMResourceEntry(
        "PaLM (Scaling Language Modeling with Pathways).pdf",
        "PaLM",
        ("palm",),
    ),
    LLMResourceEntry("Parameter-Server.pdf", "Parameter Server", ("parameter_server",)),
    LLMResourceEntry("Pathways .pdf", "Pathways", ("pathways",)),
    LLMResourceEntry("Pile.pdf", "The Pile", ("pile",)),
    LLMResourceEntry(
        "Prompt Engineering guide.pdf",
        "Prompt Engineering Guide",
        ("prompt_engineering_guide",),
    ),
    LLMResourceEntry("ScienceQA.pdf", "ScienceQA", ("scienceqa",)),
    LLMResourceEntry("Segment anything’s .pdf", "Segment Anything", ("segment_anything",)),
    LLMResourceEntry("Self- instruct.pdf", "Self-Instruct", ("self_instruct",)),
    LLMResourceEntry("The stack.pdf", "The Stack", ("the_stack",)),
    LLMResourceEntry("Ul2.pdf", "UL2", ("ul2",)),
    LLMResourceEntry("Vilt.pdf", "ViLT", ("vilt",)),
    LLMResourceEntry("Zeros.pdf", "ZeRO", ("zero",)),
    LLMResourceEntry(
        "dataset (3-5).pdf",
        "LLM Dataset Notes",
        (),
        status="reference_only",
        notes="Dataset overview notes, not a single standalone paper-shaped implementation.",
    ),
    LLMResourceEntry("google-about-bard.pdf", "Bard", ("bard",)),
    LLMResourceEntry("gpipe.pdf", "GPipe", ("gpipe",)),
    LLMResourceEntry("gpt-neox.pdf", "GPT-NeoX", ("gpt_neox",)),
    LLMResourceEntry(
        "mingpt4.pdf",
        "MiniGPT-4",
        ("minigpt4",),
        notes="Resource filename omits one 'i'; implementation keeps the canonical MiniGPT-4 spelling.",
    ),
    LLMResourceEntry("pythia.pdf", "Pythia", ("pythia",)),
    LLMResourceEntry("self-instruct.pdf", "Self-Instruct", ("self_instruct",)),
    LLMResourceEntry(
        "timeline1.pdf",
        "LLM Timeline",
        ("llm_timeline",),
        notes="Timeline resource rather than a single network paper.",
    ),
    LLMResourceEntry(
        "多模态统一框架之BLIP系列工作.pdf",
        "BLIP Series Overview",
        ("blip", "blip2", "instructblip"),
        status="implemented_family_note",
        notes="A BLIP-family note that spans multiple paper modules.",
    ),
    LLMResourceEntry(
        "大模型.md",
        "Large Model Notes",
        (),
        status="reference_only",
        notes="General notes, not a standalone paper-shaped module.",
    ),
    LLMResourceEntry("思维链（Chain-of-thoughts）.pdf", "Chain-of-Thought Prompting", ("chain_of_thought",)),
)


LLM_RESOURCE_INDEX: dict[str, LLMResourceEntry] = {entry.filename: entry for entry in _ENTRIES}

if len(LLM_RESOURCE_INDEX) != len(_ENTRIES):  # pragma: no cover
    raise RuntimeError("Duplicate filenames detected in Llms resource registry")


def list_llm_resource_entries() -> tuple[LLMResourceEntry, ...]:
    return _ENTRIES


def get_llm_resource_entry(filename: str) -> LLMResourceEntry:
    return LLM_RESOURCE_INDEX[str(filename)]


__all__ = [
    "LLMResourceEntry",
    "LLM_RESOURCE_INDEX",
    "get_llm_resource_entry",
    "list_llm_resource_entries",
]
