# Multimodal VLM Design

**Date:** 2026-03-14

**Goal:** Add a learning-first local Vision-Language Model zoo to DL-Hub with timeline-first presentation, family-based code organization, offline smoke support, and recommendation tooling.

## Problem

DL-Hub already covers strong single-modality areas such as vision, NLP, point cloud, and generative modeling, but it lacks a unified multimodal track that shows how modern image-text models evolved from dual-encoder retrieval systems into instruction-tuned multimodal LLMs. The repository also already contains several paper-shaped multimodal implementations in `Llms/`, which makes a clean, zoo-style `dlhub/` entrypoint the next logical step.

## Scope

The first version focuses on a local VLM zoo rather than a full lesson track. It includes:

- A new top-level package: `dlhub/multimodal/vlm/`
- A local registry entrypoint: `dlhub/multimodal/vlm_zoo.py`
- A CLI utility: `scripts/vlm_zoo.py`
- Timeline metadata sorted by year
- Recommendation profiles for common usage modes
- A compact-first shared VLM core with family wrappers
- 12 VLM families and 36 local architecture ids

This first version does not include dataset pipelines, real checkpoint loading, or full task-specific lessons.

## Architectural Choice

### Chosen structure

- Code layout: `dlhub/multimodal/vlm/<family>.py`
- Registry layout: `vlm:<family>_<variant>`
- Presentation layout: timeline, README, and CLI sorted by `year`

This gives the repository two views of the same space:

- Engineering view: one file per family, easy to maintain and extend
- Learning view: year-first evolution, easy to understand historically

### Why not place VLM under `vision/`

Although many VLM papers are vision-led, the models are not purely visual. They mix image encoders, text encoders, alignment objectives, fusion blocks, and multimodal generation. A dedicated `multimodal` top-level package keeps the taxonomy clean and leaves room for future directions such as document VLMs, grounding, VQA, and multimodal reasoning.

## Families

The first batch includes 12 families:

- 2021: `vilt`, `clip`, `align`, `albef`
- 2022: `ofa`, `blip`, `coca`, `flamingo`
- 2023: `blip2`, `instructblip`, `llava`, `kosmos2`

These cover four practical VLM paradigms:

- `single_stream`: early unified image-text transformer models
- `dual_encoder`: contrastive image-text retrieval/alignment models
- `fusion_encoder_decoder`: encoder-decoder or multimodal fusion models
- `multimodal_llm`: bridge-based and instruction-tuned multimodal language models

## Compact Core

The shared `CompactVLM` module will provide lightweight offline behavior for smoke tests. It will not try to replicate full paper fidelity. Instead, it will expose stable multimodal-shaped outputs:

- `image_embed`
- `text_embed`
- `logits`
- `generated_tokens` for generation-style families

The compact core will support four switches:

- `architecture_mode`: `single_stream`, `dual_encoder`, `fusion`, `bridge`
- `use_instruction`: whether prompt tokens condition the text side
- `use_query_bridge`: whether image features are routed through learned query tokens
- `use_generation_head`: whether token generation outputs are produced

## Public API

### Python API

- `dlhub.multimodal.vlm_zoo.list_local_arches()`
- `dlhub.multimodal.vlm_zoo.build_local_model()`

### CLI API

- `python scripts/vlm_zoo.py --list`
- `python scripts/vlm_zoo.py --timeline`
- `python scripts/vlm_zoo.py --list-profiles`
- `python scripts/vlm_zoo.py --recommend instruction --variant tiny --top-k 4`
- `python scripts/vlm_zoo.py --smoke vlm:clip_tiny`

## Recommendation Profiles

The first version will include five profiles:

- `balanced`
- `retrieval`
- `captioning`
- `instruction`
- `lightweight`

These will rank families using a combination of paradigm group, explicit family preference, and a modest modern-year bias.

## Testing Strategy

Tests will follow the same pattern already used by GAN, Diffusion, MOT, and tracking3d:

- `tests/test_dlhub_multimodal_vlm_zoo.py`
- `tests/test_dlhub_multimodal_vlm_timeline.py`
- `tests/test_dlhub_multimodal_vlm_algorithms.py`
- `tests/test_dlhub_multimodal_vlm_recommend.py`

Additionally, `tests/test_zoo_conventions_smoke.py` will include `dlhub/multimodal/vlm`.

## Risks and Mitigations

- Risk: VLM family taxonomy drifts into an unmaintainable mix of retrieval and generation terms.
  Mitigation: keep code grouped by family, timeline grouped by year, and recommendation grouped by paradigm.

- Risk: Tests become tightly coupled to one compact output shape.
  Mitigation: standardize only a small common contract and allow optional extra keys for generation families.

- Risk: The zoo grows faster than the timeline remains readable.
  Mitigation: use year-first CLI output and keep methods short, one-line descriptions.

## Success Criteria

The first VLM drop is complete when:

- 12 families / 36 arches are discoverable via `vlm:` ids
- builders exist for representative families
- timeline output is sorted by year and covers all families
- recommendation profiles return stable, valid local ids
- CLI smoke works offline
- all targeted tests and lint checks pass
