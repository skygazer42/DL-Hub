# Multimodal Lesson 04 Grounding-Lite Implementation Plan

**Goal:** Add `tracks/multimodal/lesson_04_grounding_compact_refexp` as a teaching lesson for text-conditioned bbox grounding with grid-cell localization and box decoding.

**Architecture:** The lesson will define its own synthetic multi-object referring-expression dataset, lesson-local vocabulary, spatial CNN backbone, lightweight text encoder, per-cell multimodal fusion, and a grounding head that predicts target cell logits plus box deltas. Training will supervise cell classification and target-cell box regression, then decode a bbox for evaluation and sample logging.

**Tech Stack:** Python, PyTorch, pytest, ruff

---

### Task 1: Lock lesson discovery and API contracts with red tests

**Files:**
- Modify: `tests/test_scripts_run_lesson.py`
- Create: `tests/test_tracks_multimodal_grounding.py`

**Step 1: Write the failing test**

Add tests that require:

- `lesson_04_grounding_compact_refexp` to appear in `python scripts/run_lesson.py multimodal --list`
- dry-run resolution to `tracks.multimodal.lesson_04_grounding_compact_refexp.train`
- batch dictionaries containing grounding fields
- forward outputs containing `cell_logits`, `box_deltas`, and `pred_boxes`
- a tiny training smoke run writing standard artifacts

**Step 2: Run test to verify it fails**

Run:

`pytest -q tests/test_scripts_run_lesson.py tests/test_tracks_multimodal_grounding.py`

Expected: FAIL because lesson 4 files do not exist yet.

**Step 3: Write minimal implementation**

Create only the package shell needed for the first failing assertion.

**Step 4: Run test to verify it still fails for the next missing behavior**

Re-run:

`pytest -q tests/test_scripts_run_lesson.py tests/test_tracks_multimodal_grounding.py`

Use the next failure as the next implementation target.

### Task 2: Build the synthetic grounding dataset

**Files:**
- Modify: `tracks/multimodal/README.md`
- Create: `tracks/multimodal/lesson_04_grounding_compact_refexp/__init__.py`
- Create: `tracks/multimodal/lesson_04_grounding_compact_refexp/data.py`
- Create: `tracks/multimodal/lesson_04_grounding_compact_refexp/README.md`
- Test: `tests/test_tracks_multimodal_grounding.py`

**Step 1: Write the failing test**

Add a test that asserts:

- image batch shape is `(B, 3, H, W)`
- `input_ids` and `attention_mask` have consistent shapes
- `target_cell` is shaped `(B,)`
- `target_box` is shaped `(B, 4)`
- the vocabulary includes grounding words such as `find`, `locate`, `object`, `top`, `left`

**Step 2: Run test to verify it fails**

Run:

`pytest -q tests/test_tracks_multimodal_grounding.py::test_multimodal_grounding_batch_shapes`

Expected: FAIL because `data.py` is missing.

**Step 3: Write minimal implementation**

Implement:

- lesson-local vocabulary
- synthetic multi-object scene generation
- unique referring-expression construction
- target grid and delta computation
- collate and dataloader helpers

**Step 4: Run test to verify it passes**

Run the same test and expect PASS.

### Task 3: Implement the grounding model

**Files:**
- Create: `tracks/multimodal/lesson_04_grounding_compact_refexp/model.py`
- Test: `tests/test_tracks_multimodal_grounding.py`

**Step 1: Write the failing test**

Add tests that require:

- `cell_logits` shaped `(B, num_cells)`
- `box_deltas` shaped `(B, num_cells, 4)`
- decoded `pred_boxes` shaped `(B, 4)`
- finite grounding loss

**Step 2: Run test to verify it fails**

Run:

`pytest -q tests/test_tracks_multimodal_grounding.py::test_multimodal_grounding_model_outputs`

Expected: FAIL because `model.py` is missing.

**Step 3: Write minimal implementation**

Implement:

- `VisionEncoder`
- `TextEncoder`
- per-cell fusion
- `GroundingHead`
- bbox decode and grounding loss helpers

**Step 4: Run test to verify it passes**

Run the same test and expect PASS.

### Task 4: Add the training entrypoint and smoke run

**Files:**
- Create: `tracks/multimodal/lesson_04_grounding_compact_refexp/train.py`
- Test: `tests/test_tracks_multimodal_grounding.py`

**Step 1: Write the failing test**

Add a subprocess smoke test that runs:

`python -m tracks.multimodal.lesson_04_grounding_compact_refexp.train --epochs 1 --num-samples 64 --batch-size 8 --image-size 32 --grid-size 4 --max-train-batches 2 --max-eval-batches 1 --device cpu --run-name pytest_grounding_smoke`

Assert that it exits successfully and writes:

- `outputs/multimodal/lesson_04_grounding_compact_refexp/pytest_grounding_smoke/config.json`
- `outputs/multimodal/lesson_04_grounding_compact_refexp/pytest_grounding_smoke/metrics.jsonl`
- `outputs/multimodal/lesson_04_grounding_compact_refexp/pytest_grounding_smoke/checkpoints/checkpoint.pt`

**Step 2: Run test to verify it fails**

Run:

`pytest -q tests/test_tracks_multimodal_grounding.py::test_multimodal_grounding_training_smoke`

Expected: FAIL because `train.py` does not exist yet.

**Step 3: Write minimal implementation**

Implement:

- CLI argument parsing
- grounding training loop
- metric computation for cell accuracy, bbox L1, and center accuracy
- config, vocab, metrics, sample logging, and checkpoint writing

**Step 4: Run test to verify it passes**

Run the same test and expect PASS.

### Task 5: Verify the feature

**Files:**
- Modify: `tests/test_scripts_run_lesson.py`
- Create: `tests/test_tracks_multimodal_grounding.py`
- Modify: `tracks/multimodal/README.md`
- Create: `tracks/multimodal/lesson_04_grounding_compact_refexp/__init__.py`
- Create: `tracks/multimodal/lesson_04_grounding_compact_refexp/data.py`
- Create: `tracks/multimodal/lesson_04_grounding_compact_refexp/model.py`
- Create: `tracks/multimodal/lesson_04_grounding_compact_refexp/train.py`
- Create: `tracks/multimodal/lesson_04_grounding_compact_refexp/README.md`

**Step 1: Run lint**

Run:

`ruff check tracks/multimodal tests/test_tracks_multimodal_grounding.py tests/test_tracks_multimodal_llava.py tests/test_tracks_multimodal_blip.py tests/test_tracks_multimodal_clip.py tests/test_scripts_run_lesson.py`

Expected: PASS.

**Step 2: Run targeted tests**

Run:

`pytest -q tests/test_scripts_run_lesson.py tests/test_tracks_multimodal_clip.py tests/test_tracks_multimodal_blip.py tests/test_tracks_multimodal_llava.py tests/test_tracks_multimodal_grounding.py`

Expected: PASS.

**Step 3: Run manual discovery smoke**

Run:

- `python scripts/run_lesson.py multimodal --list`
- `python scripts/run_lesson.py multimodal lesson_04_grounding_compact_refexp --dry-run`

Expected: lesson 4 appears in the track listing and resolves to the train module.
