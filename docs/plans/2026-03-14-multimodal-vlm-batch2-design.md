# Multimodal VLM Batch 2 Design

**Date:** 2026-03-14

**Goal:** Extend the existing local VLM zoo with a second batch of 8 core families so the timeline better covers the evolution from retrieval-style VLMs to stronger multimodal LLMs.

## Scope

This batch adds:

- `simvlm`
- `lit`
- `pali`
- `pali_x`
- `minigpt4`
- `mplug_owl2`
- `qwen_vl`
- `cogvlm`

The resulting VLM zoo grows from 12 families / 36 arches to 20 families / 60 arches.

## Rationale

The first VLM batch established the basic taxonomy:

- early single-stream models
- dual-encoder alignment models
- fusion encoder-decoder models
- bridge-style multimodal LLMs

The second batch should not invent a new structure. It should deepen the existing one:

- add missing 2021-2022 retrieval/generative milestone families
- strengthen the 2023 multimodal LLM surge
- keep the timeline presentation chronological
- keep code layout stable and family-based

## Timeline Placement

- 2021: `simvlm`, `lit`
- 2022: `pali`
- 2023: `pali_x`, `minigpt4`, `mplug_owl2`, `qwen_vl`, `cogvlm`

## Group Mapping

- `lit` -> `dual_encoder`
- `simvlm` -> `fusion_encoder_decoder`
- `pali` -> `fusion_encoder_decoder`
- `pali_x` -> `fusion_encoder_decoder`
- `minigpt4` -> `multimodal_llm`
- `mplug_owl2` -> `multimodal_llm`
- `qwen_vl` -> `multimodal_llm`
- `cogvlm` -> `multimodal_llm`

This preserves the original four-group taxonomy and avoids exploding the recommendation surface with over-specific subgroups.

## Compact Implementation Shape

The shared `CompactVLM` core remains sufficient. The second batch only needs new family wrappers and modest scoring/timeline updates:

- `lit` uses the `dual_encoder` path
- `simvlm`, `pali`, `pali_x` use the generative fusion path
- `minigpt4`, `mplug_owl2`, `qwen_vl`, `cogvlm` use the bridge-style multimodal LLM path

Where useful, instruction-aware families can enable `use_instruction=True`.

## Recommendation Implications

- `retrieval` should start preferring `lit`
- `captioning` should start preferring `simvlm` and `pali`
- `instruction` should surface `qwen_vl`, `mplug_owl2`, `cogvlm`, and `minigpt4`
- `balanced` should mix old and new families instead of biasing too heavily to the first batch

## Success Criteria

This batch is complete when:

- `list_local_arches()` returns at least 60 `vlm:` ids
- timeline metadata covers at least 20 families
- representative ids from the new families are buildable
- recommendation results can return new-batch families
- `scripts/vlm_zoo.py` continues to work without interface changes
