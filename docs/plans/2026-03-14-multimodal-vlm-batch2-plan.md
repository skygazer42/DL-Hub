# Multimodal VLM Batch 2 Implementation Plan

**Goal:** Extend the local VLM zoo with 8 additional core families, growing it to 20 families / 60 arches while preserving the year-first timeline and existing CLI.

**Architecture:** Reuse the current `dlhub.multimodal.vlm` structure, add one wrapper module per new family, extend `_timeline.py` and `_recommend.py`, and keep the shared `CompactVLM` core unchanged except where small compatibility tweaks are necessary. The registry and CLI should discover new families automatically through the existing per-file `_VARIANTS` + `build_<family>_vlm(...)` pattern.

**Tech Stack:** Python, PyTorch, pytest, ruff

---

### Task 1: Expand the VLM tests to require the second batch

**Files:**
- Modify: `tests/test_dlhub_multimodal_vlm_zoo.py`
- Modify: `tests/test_dlhub_multimodal_vlm_timeline.py`
- Modify: `tests/test_dlhub_multimodal_vlm_algorithms.py`
- Modify: `tests/test_dlhub_multimodal_vlm_recommend.py`

**Step 1: Write the failing test**

Require:

- at least 60 `vlm:` arches
- representative ids for `simvlm`, `lit`, `pali`, `qwen_vl`, `cogvlm`
- timeline size >= 20
- correct years/groups for the new families
- recommendation behavior that can return new instruction-oriented families

**Step 2: Run test to verify it fails**

Run: `pytest -q tests/test_dlhub_multimodal_vlm_zoo.py tests/test_dlhub_multimodal_vlm_timeline.py tests/test_dlhub_multimodal_vlm_algorithms.py tests/test_dlhub_multimodal_vlm_recommend.py`

Expected: FAIL because the second-batch families do not exist yet.

**Step 3: Write minimal implementation**

Add only the minimum files and metadata needed to satisfy the new failing assertions.

**Step 4: Run test to verify it passes**

Re-run the same command and continue only once all second-batch VLM tests pass.

### Task 2: Add the 8 new family wrappers

**Files:**
- Create: `dlhub/multimodal/vlm/simvlm.py`
- Create: `dlhub/multimodal/vlm/lit.py`
- Create: `dlhub/multimodal/vlm/pali.py`
- Create: `dlhub/multimodal/vlm/pali_x.py`
- Create: `dlhub/multimodal/vlm/minigpt4.py`
- Create: `dlhub/multimodal/vlm/mplug_owl2.py`
- Create: `dlhub/multimodal/vlm/qwen_vl.py`
- Create: `dlhub/multimodal/vlm/cogvlm.py`

**Step 1: Write the failing test**

Use the algorithm smoke tests as the contract for builder names and output shapes.

**Step 2: Run test to verify it fails**

Run: `pytest -q tests/test_dlhub_multimodal_vlm_algorithms.py`

Expected: FAIL with missing builder/module errors for the new families.

**Step 3: Write minimal implementation**

Use the existing `CompactVLM` family-builder helper and only vary:

- `_VARIANTS`
- `architecture_mode`
- `use_instruction`
- `use_query_bridge`
- `use_generation_head`

**Step 4: Run test to verify it passes**

Run the same command and expect all representative builders to pass.

### Task 3: Extend timeline, recommendation, and docs

**Files:**
- Modify: `dlhub/multimodal/vlm/_timeline.py`
- Modify: `dlhub/multimodal/vlm/_recommend.py`
- Modify: `dlhub/multimodal/vlm/README.md`

**Step 1: Write the failing test**

Use timeline and recommend tests as the contract for the new families.

**Step 2: Run test to verify it fails**

Run: `pytest -q tests/test_dlhub_multimodal_vlm_timeline.py tests/test_dlhub_multimodal_vlm_recommend.py`

Expected: FAIL because metadata and ranking logic still describe only the first batch.

**Step 3: Write minimal implementation**

Add the new families to:

- `_timeline.py`
- recommendation preferences
- README family inventory

**Step 4: Run test to verify it passes**

Run the same command and expect updated timeline and recommendation checks to pass.

### Task 4: Verify the expanded zoo end-to-end

**Files:**
- Modify: `scripts/vlm_zoo.py` only if needed for display parity
- Test: `tests/test_dlhub_multimodal_vlm_zoo.py`
- Test: `tests/test_dlhub_multimodal_vlm_timeline.py`
- Test: `tests/test_dlhub_multimodal_vlm_algorithms.py`
- Test: `tests/test_dlhub_multimodal_vlm_recommend.py`

**Step 1: Run lint**

Run: `ruff check dlhub/multimodal dlhub/multimodal/vlm_zoo.py scripts/vlm_zoo.py tests/test_dlhub_multimodal_vlm_zoo.py tests/test_dlhub_multimodal_vlm_timeline.py tests/test_dlhub_multimodal_vlm_algorithms.py tests/test_dlhub_multimodal_vlm_recommend.py`

Expected: PASS.

**Step 2: Run targeted tests**

Run: `pytest -q tests/test_dlhub_multimodal_vlm_zoo.py tests/test_dlhub_multimodal_vlm_timeline.py tests/test_dlhub_multimodal_vlm_algorithms.py tests/test_dlhub_multimodal_vlm_recommend.py tests/test_zoo_conventions_smoke.py`

Expected: PASS.

**Step 3: Run manual CLI smoke**

Run:

- `python scripts/vlm_zoo.py --list --limit 8`
- `python scripts/vlm_zoo.py --timeline`
- `python scripts/vlm_zoo.py --recommend instruction --variant tiny --top-k 6`
- `python scripts/vlm_zoo.py --smoke vlm:qwen_vl_tiny`

Expected: the CLI lists at least 60 arches and shows the new families in timeline and recommendation output.
