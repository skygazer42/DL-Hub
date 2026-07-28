# Multimodal VLM Zoo Implementation Plan

**Goal:** Build a local multimodal VLM zoo with 12 families, 36 local arches, year-first timeline tooling, recommendation profiles, and an offline smoke CLI.

**Architecture:** Add a new top-level `dlhub.multimodal` package, implement a shared compact VLM core in `dlhub/multimodal/vlm/_common.py`, keep one family per file under `dlhub/multimodal/vlm/`, and expose the registry through `dlhub/multimodal/vlm_zoo.py`. Present the space by year in timeline and README while keeping code organized by family for maintainability.

**Tech Stack:** Python, PyTorch, pytest, ruff

---

### Task 1: Lock the red test contract for the VLM zoo

**Files:**
- Create: `tests/test_dlhub_multimodal_vlm_zoo.py`
- Create: `tests/test_dlhub_multimodal_vlm_timeline.py`
- Create: `tests/test_dlhub_multimodal_vlm_algorithms.py`
- Create: `tests/test_dlhub_multimodal_vlm_recommend.py`
- Modify: `tests/test_zoo_conventions_smoke.py`

**Step 1: Write the failing test**

Add tests that require:

- at least 36 local arches
- representative ids such as `vlm:clip_tiny`, `vlm:blip_small`, `vlm:llava_base`
- builders for representative families
- timeline coverage for 2021, 2022, 2023
- recommendation profiles for retrieval, captioning, instruction, and lightweight use cases

**Step 2: Run test to verify it fails**

Run: `pytest -q tests/test_dlhub_multimodal_vlm_zoo.py tests/test_dlhub_multimodal_vlm_timeline.py tests/test_dlhub_multimodal_vlm_algorithms.py tests/test_dlhub_multimodal_vlm_recommend.py`

Expected: FAIL because `dlhub.multimodal` and `scripts/vlm_zoo.py` do not exist yet.

**Step 3: Write minimal implementation**

Create the package skeleton and empty stubs only after the red test output is confirmed.

**Step 4: Run test to verify it still fails for the next missing behavior**

Re-run the same `pytest` command and use the next failure as the implementation target.

### Task 2: Build the shared VLM core and the 12 family wrappers

**Files:**
- Create: `dlhub/multimodal/__init__.py`
- Create: `dlhub/multimodal/vlm/__init__.py`
- Create: `dlhub/multimodal/vlm/_common.py`
- Create: `dlhub/multimodal/vlm/vilt.py`
- Create: `dlhub/multimodal/vlm/clip.py`
- Create: `dlhub/multimodal/vlm/align.py`
- Create: `dlhub/multimodal/vlm/albef.py`
- Create: `dlhub/multimodal/vlm/ofa.py`
- Create: `dlhub/multimodal/vlm/blip.py`
- Create: `dlhub/multimodal/vlm/coca.py`
- Create: `dlhub/multimodal/vlm/flamingo.py`
- Create: `dlhub/multimodal/vlm/blip2.py`
- Create: `dlhub/multimodal/vlm/instructblip.py`
- Create: `dlhub/multimodal/vlm/llava.py`
- Create: `dlhub/multimodal/vlm/kosmos2.py`

**Step 1: Write the failing test**

Use the algorithm smoke tests and the conventions smoke test as the contract for:

- `_VARIANTS`
- `build_<family>_vlm(...)`
- `if __name__ == "__main__":`
- stable multimodal output keys

**Step 2: Run test to verify it fails**

Run: `pytest -q tests/test_dlhub_multimodal_vlm_algorithms.py tests/test_zoo_conventions_smoke.py`

Expected: FAIL because the family builders and modules are still missing.

**Step 3: Write minimal implementation**

Implement a shared `CompactVLM` that returns:

- `image_embed`
- `text_embed`
- `logits`
- optional `generated_tokens`

Keep each family wrapper thin and only set family-specific mode flags.

**Step 4: Run test to verify it passes**

Run the same `pytest` command and expect the family API and conventions to pass.

### Task 3: Add timeline, recommender, registry, README, and CLI

**Files:**
- Create: `dlhub/multimodal/vlm/_timeline.py`
- Create: `dlhub/multimodal/vlm/_recommend.py`
- Create: `dlhub/multimodal/vlm/README.md`
- Create: `dlhub/multimodal/vlm_zoo.py`
- Create: `scripts/vlm_zoo.py`

**Step 1: Write the failing test**

Use the timeline and recommendation tests as the contract for year ordering, profile behavior, and CLI text.

**Step 2: Run test to verify it fails**

Run: `pytest -q tests/test_dlhub_multimodal_vlm_timeline.py tests/test_dlhub_multimodal_vlm_recommend.py`

Expected: FAIL because timeline metadata and recommendation logic are missing.

**Step 3: Write minimal implementation**

Add:

- timeline entries for 2021, 2022, 2023
- recommendation profiles `balanced`, `retrieval`, `captioning`, `instruction`, `lightweight`
- `vlm:` registry output
- CLI commands `--list`, `--timeline`, `--list-profiles`, `--recommend`, `--smoke`

**Step 4: Run test to verify it passes**

Run the same `pytest` command and expect metadata and CLI coverage to pass.

### Task 4: Verify the whole feature

**Files:**
- Test: `tests/test_dlhub_multimodal_vlm_zoo.py`
- Test: `tests/test_dlhub_multimodal_vlm_timeline.py`
- Test: `tests/test_dlhub_multimodal_vlm_algorithms.py`
- Test: `tests/test_dlhub_multimodal_vlm_recommend.py`
- Modify: `tests/test_zoo_conventions_smoke.py`

**Step 1: Run lint**

Run: `ruff check dlhub/multimodal dlhub/multimodal/vlm_zoo.py scripts/vlm_zoo.py tests/test_dlhub_multimodal_vlm_zoo.py tests/test_dlhub_multimodal_vlm_timeline.py tests/test_dlhub_multimodal_vlm_algorithms.py tests/test_dlhub_multimodal_vlm_recommend.py tests/test_zoo_conventions_smoke.py`

Expected: PASS.

**Step 2: Run targeted tests**

Run: `pytest -q tests/test_dlhub_multimodal_vlm_zoo.py tests/test_dlhub_multimodal_vlm_timeline.py tests/test_dlhub_multimodal_vlm_algorithms.py tests/test_dlhub_multimodal_vlm_recommend.py tests/test_zoo_conventions_smoke.py`

Expected: PASS.

**Step 3: Run manual CLI smoke**

Run:

- `python scripts/vlm_zoo.py --list --limit 6`
- `python scripts/vlm_zoo.py --timeline`
- `python scripts/vlm_zoo.py --recommend instruction --variant tiny --top-k 4`
- `python scripts/vlm_zoo.py --smoke vlm:clip_tiny`

Expected: readable VLM output with `vlm:` ids and year-first timeline sections.
